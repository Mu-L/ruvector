//! Path validation utilities
//!
//! Provides protection against path traversal attacks by validating
//! that paths stay within allowed directories.

use crate::error::{SecurityError, SecurityResult};
use std::path::{Path, PathBuf};

/// Path validator that ensures paths stay within allowed directories
///
/// # Example
///
/// ```rust,no_run
/// use ruvector_security::PathValidator;
/// use std::path::PathBuf;
///
/// // Create validator with allowed directories
/// let validator = PathValidator::new(vec![PathBuf::from("/data"), PathBuf::from("/tmp")]);
///
/// // Paths within allowed directories are valid
/// // (Note: these paths must exist for validate() to succeed)
/// // validator.validate("/data/vectors.db");
///
/// // Paths outside allowed directories are rejected
/// assert!(validator.validate("/etc/passwd").is_err());
///
/// // Path traversal attempts are rejected
/// assert!(validator.validate("/data/../etc/passwd").is_err());
/// ```
#[derive(Debug, Clone)]
pub struct PathValidator {
    /// Allowed directories (canonicalized)
    allowed_dirs: Vec<PathBuf>,
    /// Whether to allow symlinks
    allow_symlinks: bool,
    /// Maximum path length
    max_path_length: usize,
}

impl PathValidator {
    /// Maximum allowed path length (default 4096)
    pub const DEFAULT_MAX_PATH_LENGTH: usize = 4096;

    /// Create a new path validator with allowed directories
    pub fn new(allowed_dirs: Vec<PathBuf>) -> Self {
        // Canonicalize allowed directories where possible
        let allowed_dirs = allowed_dirs
            .into_iter()
            .filter_map(|p| {
                p.canonicalize().ok().or_else(|| {
                    // If canonicalization fails, try to resolve relative to cwd
                    std::env::current_dir()
                        .ok()
                        .map(|cwd| cwd.join(&p))
                        .and_then(|p| p.canonicalize().ok())
                        .or(Some(p))
                })
            })
            .collect();

        Self {
            allowed_dirs,
            allow_symlinks: false,
            max_path_length: Self::DEFAULT_MAX_PATH_LENGTH,
        }
    }

    /// Set whether symlinks are allowed
    pub fn allow_symlinks(mut self, allow: bool) -> Self {
        self.allow_symlinks = allow;
        self
    }

    /// Set maximum path length
    pub fn max_path_length(mut self, length: usize) -> Self {
        self.max_path_length = length;
        self
    }

    /// Validate a path and return the canonical path if valid
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path contains path traversal sequences (`..`)
    /// - The path is outside all allowed directories
    /// - The path is a symlink (unless allowed)
    /// - The path exceeds maximum length
    pub fn validate<P: AsRef<Path>>(&self, path: P) -> SecurityResult<PathBuf> {
        let path = path.as_ref();
        let path_str = path.to_string_lossy();

        // Check path length
        if path_str.len() > self.max_path_length {
            return Err(SecurityError::InvalidPath(path.to_path_buf()));
        }

        // Check for null bytes (security risk)
        if path_str.contains('\0') {
            return Err(SecurityError::InvalidPathCharacters(path.to_path_buf()));
        }

        // Check for explicit path traversal in string form
        // This catches attempts like "foo/../../etc/passwd"
        if path_str.contains("..") {
            // Additional check: see if normalized path still contains traversal
            let normalized = self.normalize_path(path)?;
            // If after normalization it differs significantly, it's suspicious
            if !self.is_path_safe(&normalized)? {
                return Err(SecurityError::PathTraversal(path.to_path_buf()));
            }
        }

        // Resolve to canonical path
        let canonical = self.resolve_canonical(path)?;

        // Check symlink if not allowed
        if !self.allow_symlinks {
            if let Ok(metadata) = std::fs::symlink_metadata(&canonical) {
                if metadata.file_type().is_symlink() {
                    return Err(SecurityError::SymlinkDetected(path.to_path_buf()));
                }
            }
            // Also check the original path for symlinks
            if let Ok(metadata) = std::fs::symlink_metadata(path) {
                if metadata.file_type().is_symlink() {
                    return Err(SecurityError::SymlinkDetected(path.to_path_buf()));
                }
            }
        }

        // Verify path is within allowed directories
        self.check_allowed(&canonical)?;

        Ok(canonical)
    }

    /// Validate a path for a new file (may not exist yet)
    ///
    /// This validates the parent directory exists and is within allowed paths,
    /// and that the filename is safe.
    pub fn validate_new_file<P: AsRef<Path>>(&self, path: P) -> SecurityResult<PathBuf> {
        let path = path.as_ref();
        let path_str = path.to_string_lossy();

        // Check path length
        if path_str.len() > self.max_path_length {
            return Err(SecurityError::InvalidPath(path.to_path_buf()));
        }

        // Check for null bytes
        if path_str.contains('\0') {
            return Err(SecurityError::InvalidPathCharacters(path.to_path_buf()));
        }

        // Check for path traversal
        if path_str.contains("..") {
            return Err(SecurityError::PathTraversal(path.to_path_buf()));
        }

        // Get and validate parent directory
        let parent = path.parent().ok_or_else(|| SecurityError::InvalidPath(path.to_path_buf()))?;

        let canonical_parent = if parent.exists() {
            parent.canonicalize().map_err(|_| SecurityError::InvalidPath(parent.to_path_buf()))?
        } else {
            // For new directories, resolve as much as possible
            self.resolve_existing_ancestor(parent)?
        };

        // Verify parent is within allowed directories
        self.check_allowed(&canonical_parent)?;

        // Get filename and validate it
        let filename = path
            .file_name()
            .ok_or_else(|| SecurityError::InvalidPath(path.to_path_buf()))?;

        let filename_str = filename.to_string_lossy();

        // Check for hidden files starting with dots (optional, can be removed if needed)
        // Check for dangerous characters in filename
        if filename_str.contains('/') || filename_str.contains('\\') || filename_str.contains('\0')
        {
            return Err(SecurityError::InvalidPathCharacters(path.to_path_buf()));
        }

        Ok(canonical_parent.join(filename))
    }

    /// Normalize a path by resolving `.` and `..` components
    fn normalize_path(&self, path: &Path) -> SecurityResult<PathBuf> {
        let mut normalized = PathBuf::new();

        for component in path.components() {
            match component {
                std::path::Component::ParentDir => {
                    if !normalized.pop() {
                        // Can't go above root
                        return Err(SecurityError::PathTraversal(path.to_path_buf()));
                    }
                }
                std::path::Component::CurDir => {
                    // Skip `.`
                }
                c => {
                    normalized.push(c);
                }
            }
        }

        Ok(normalized)
    }

    /// Check if a normalized path is safe (no traversal above allowed roots)
    fn is_path_safe(&self, normalized: &Path) -> SecurityResult<bool> {
        // If path exists, canonicalize and check
        if normalized.exists() {
            let canonical = normalized
                .canonicalize()
                .map_err(|_| SecurityError::InvalidPath(normalized.to_path_buf()))?;
            return Ok(self.allowed_dirs.iter().any(|dir| canonical.starts_with(dir)));
        }

        // For non-existent paths, find the nearest existing ancestor
        let mut current = normalized.to_path_buf();
        while !current.exists() {
            if !current.pop() {
                return Ok(false);
            }
        }

        if current.as_os_str().is_empty() {
            current = std::env::current_dir().map_err(SecurityError::Io)?;
        }

        let canonical = current
            .canonicalize()
            .map_err(|_| SecurityError::InvalidPath(normalized.to_path_buf()))?;

        Ok(self.allowed_dirs.iter().any(|dir| canonical.starts_with(dir)))
    }

    /// Resolve to canonical path, handling non-existent files
    fn resolve_canonical(&self, path: &Path) -> SecurityResult<PathBuf> {
        if path.exists() {
            path.canonicalize()
                .map_err(|_| SecurityError::InvalidPath(path.to_path_buf()))
        } else {
            // For non-existent paths, canonicalize the parent
            self.validate_new_file(path)
        }
    }

    /// Find the nearest existing ancestor and canonicalize it
    fn resolve_existing_ancestor(&self, path: &Path) -> SecurityResult<PathBuf> {
        let mut current = path.to_path_buf();

        while !current.exists() {
            if !current.pop() {
                // Reached root without finding existing path
                return std::env::current_dir().map_err(SecurityError::Io);
            }
        }

        if current.as_os_str().is_empty() {
            current = std::env::current_dir().map_err(SecurityError::Io)?;
        }

        current
            .canonicalize()
            .map_err(|_| SecurityError::InvalidPath(path.to_path_buf()))
    }

    /// Check if path is within allowed directories
    fn check_allowed(&self, canonical: &Path) -> SecurityResult<()> {
        let allowed = self
            .allowed_dirs
            .iter()
            .any(|dir| canonical.starts_with(dir));

        if !allowed {
            return Err(SecurityError::PathOutsideAllowed {
                path: canonical.to_path_buf(),
                allowed: self.allowed_dirs.clone(),
            });
        }

        Ok(())
    }
}

impl Default for PathValidator {
    fn default() -> Self {
        // Default to current working directory
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self::new(vec![cwd])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_path_traversal_blocked() {
        let temp = TempDir::new().unwrap();
        let validator = PathValidator::new(vec![temp.path().to_path_buf()]);

        // Path traversal should be blocked
        let evil_path = temp.path().join("../../../etc/passwd");
        assert!(validator.validate(&evil_path).is_err());
    }

    #[test]
    fn test_valid_path_allowed() {
        let temp = TempDir::new().unwrap();
        let valid_file = temp.path().join("test.db");
        std::fs::write(&valid_file, "test").unwrap();

        let validator = PathValidator::new(vec![temp.path().to_path_buf()]);
        assert!(validator.validate(&valid_file).is_ok());
    }

    #[test]
    fn test_outside_allowed_blocked() {
        let temp = TempDir::new().unwrap();
        let validator = PathValidator::new(vec![temp.path().to_path_buf()]);

        // Absolute path outside allowed directories
        let outside_path = PathBuf::from("/etc/passwd");
        assert!(validator.validate(&outside_path).is_err());
    }

    #[test]
    fn test_null_bytes_blocked() {
        let temp = TempDir::new().unwrap();
        let validator = PathValidator::new(vec![temp.path().to_path_buf()]);

        let evil_path = temp.path().join("test\0.db");
        assert!(validator.validate(&evil_path).is_err());
    }

    #[test]
    fn test_new_file_validation() {
        let temp = TempDir::new().unwrap();
        let validator = PathValidator::new(vec![temp.path().to_path_buf()]);

        // New file in valid directory
        let new_file = temp.path().join("new_vectors.db");
        assert!(validator.validate_new_file(&new_file).is_ok());

        // New file with traversal
        let evil_new = temp.path().join("../evil.db");
        assert!(validator.validate_new_file(&evil_new).is_err());
    }
}
