//! FFI safety utilities
//!
//! Provides safe wrappers for unsafe FFI operations including:
//! - Pointer validation
//! - Tracked allocations with safe deallocation
//! - SAFETY documentation patterns

use crate::error::{SecurityError, SecurityResult};
use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Maximum buffer size for FFI operations (256 MB)
pub const MAX_FFI_BUFFER_SIZE: usize = 256 * 1024 * 1024;

/// Error type for FFI operations
pub type FfiError = SecurityError;

/// Validates a raw pointer before use in unsafe code.
///
/// # Arguments
/// * `ptr` - Pointer to validate
/// * `len` - Number of elements (not bytes)
///
/// # Returns
/// * `Ok(())` if the pointer is valid
/// * `Err(FfiError)` if validation fails
///
/// # Example
///
/// ```rust,ignore
/// use ruvector_security::ffi::validate_ptr;
///
/// let data = vec![1.0f32, 2.0, 3.0];
/// let ptr = data.as_ptr();
///
/// // Valid pointer
/// assert!(validate_ptr(ptr, data.len()).is_ok());
///
/// // Null pointer
/// let null: *const f32 = std::ptr::null();
/// assert!(validate_ptr(null, 1).is_err());
/// ```
#[inline]
pub fn validate_ptr<T>(ptr: *const T, len: usize) -> SecurityResult<()> {
    // Null check
    if ptr.is_null() {
        return Err(SecurityError::NullPointer);
    }

    // Alignment check
    let alignment = std::mem::align_of::<T>();
    if (ptr as usize) % alignment != 0 {
        return Err(SecurityError::MisalignedPointer {
            ptr: ptr as usize,
            required_alignment: alignment,
        });
    }

    // Size overflow check
    let element_size = std::mem::size_of::<T>();
    let byte_len = len.checked_mul(element_size).ok_or(SecurityError::SizeOverflow)?;

    // Reasonable size bounds
    if byte_len > MAX_FFI_BUFFER_SIZE {
        return Err(SecurityError::BufferTooLarge(byte_len));
    }

    Ok(())
}

/// Validates a mutable raw pointer before use.
#[inline]
pub fn validate_ptr_mut<T>(ptr: *mut T, len: usize) -> SecurityResult<()> {
    validate_ptr(ptr as *const T, len)
}

/// Safely creates a slice from a raw pointer after validation.
///
/// # Safety
///
/// This function is unsafe because:
/// - The caller must ensure the pointer points to valid memory
/// - The memory must not be modified during the lifetime of the returned slice
/// - The memory must be valid for the entire length
///
/// # Example
///
/// ```rust,ignore
/// use ruvector_security::ffi::slice_from_ptr;
///
/// let data = vec![1i16, 2, 3, 4];
/// let ptr = data.as_ptr();
///
/// // SAFETY: We own `data` and know the pointer is valid
/// let slice = unsafe { slice_from_ptr(ptr, data.len())? };
/// assert_eq!(slice, &[1, 2, 3, 4]);
/// ```
///
/// # Errors
///
/// Returns an error if pointer validation fails.
#[inline]
pub unsafe fn slice_from_ptr<'a, T>(ptr: *const T, len: usize) -> SecurityResult<&'a [T]> {
    validate_ptr(ptr, len)?;

    // SAFETY: We've validated the pointer is:
    // - Non-null
    // - Properly aligned for T
    // - Within reasonable size bounds
    // The caller guarantees the memory is valid and won't be mutated.
    Ok(std::slice::from_raw_parts(ptr, len))
}

/// Safely creates a mutable slice from a raw pointer after validation.
///
/// # Safety
///
/// Same requirements as `slice_from_ptr`, plus:
/// - The caller must have exclusive access to the memory
#[inline]
pub unsafe fn slice_from_ptr_mut<'a, T>(ptr: *mut T, len: usize) -> SecurityResult<&'a mut [T]> {
    validate_ptr_mut(ptr, len)?;

    // SAFETY: We've validated the pointer, and caller guarantees exclusive access.
    Ok(std::slice::from_raw_parts_mut(ptr, len))
}

/// A tracked allocation that stores its layout for safe deallocation.
///
/// This wrapper ensures that memory is deallocated with the same layout
/// that was used for allocation, preventing undefined behavior.
///
/// # Example
///
/// ```rust
/// use ruvector_security::TrackedAllocation;
///
/// // Allocate memory for 100 f32 values
/// let mut alloc = TrackedAllocation::<f32>::new(100).unwrap();
///
/// // Write data
/// unsafe {
///     for i in 0..100 {
///         *alloc.as_mut_ptr().add(i) = i as f32;
///     }
/// }
///
/// // Memory is automatically deallocated when `alloc` is dropped
/// ```
pub struct TrackedAllocation<T> {
    /// Non-null pointer to allocated memory
    ptr: NonNull<T>,
    /// Layout used for allocation
    layout: Layout,
    /// Number of elements
    len: usize,
    /// Marker for drop check
    _marker: PhantomData<T>,
}

impl<T> TrackedAllocation<T> {
    /// Allocate memory for `count` elements of type T.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `count` is zero
    /// - The allocation size overflows
    /// - The allocator fails
    pub fn new(count: usize) -> SecurityResult<Self> {
        if count == 0 {
            return Err(SecurityError::SizeOverflow);
        }

        let layout = Layout::array::<T>(count).map_err(|_| SecurityError::SizeOverflow)?;

        if layout.size() > MAX_FFI_BUFFER_SIZE {
            return Err(SecurityError::BufferTooLarge(layout.size()));
        }

        // SAFETY: We've validated the layout is valid and within bounds.
        let ptr = unsafe { alloc(layout) as *mut T };

        let ptr = NonNull::new(ptr).ok_or(SecurityError::AllocationFailed)?;

        Ok(Self {
            ptr,
            layout,
            len: count,
            _marker: PhantomData,
        })
    }

    /// Allocate and zero-initialize memory.
    pub fn new_zeroed(count: usize) -> SecurityResult<Self> {
        let alloc = Self::new(count)?;

        // SAFETY: We just allocated this memory and have exclusive access.
        unsafe {
            std::ptr::write_bytes(alloc.ptr.as_ptr(), 0, count);
        }

        Ok(alloc)
    }

    /// Get a raw pointer to the allocation.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get a mutable raw pointer to the allocation.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the allocation is empty (always false for valid allocations).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the layout used for allocation.
    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Convert to a slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure the memory has been properly initialized.
    #[inline]
    pub unsafe fn as_slice(&self) -> &[T] {
        std::slice::from_raw_parts(self.ptr.as_ptr(), self.len)
    }

    /// Convert to a mutable slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure the memory has been properly initialized.
    #[inline]
    pub unsafe fn as_slice_mut(&mut self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
    }

    /// Copy data from a slice into the allocation.
    ///
    /// # Panics
    ///
    /// Panics if the slice length doesn't match the allocation length.
    pub fn copy_from_slice(&mut self, src: &[T])
    where
        T: Copy,
    {
        assert_eq!(
            src.len(),
            self.len,
            "Source slice length must match allocation length"
        );

        // SAFETY: We have exclusive access and lengths match.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.as_ptr(), self.len);
        }
    }

    /// Take ownership and return raw parts.
    ///
    /// After calling this, the caller is responsible for deallocation.
    /// Use `TrackedAllocation::from_raw_parts` to reconstruct.
    pub fn into_raw_parts(self) -> (*mut T, Layout, usize) {
        let parts = (self.ptr.as_ptr(), self.layout, self.len);
        std::mem::forget(self); // Prevent Drop from running
        parts
    }

    /// Reconstruct from raw parts.
    ///
    /// # Safety
    ///
    /// The caller must provide parts from a previous `into_raw_parts` call
    /// with matching type T.
    pub unsafe fn from_raw_parts(ptr: *mut T, layout: Layout, len: usize) -> Option<Self> {
        NonNull::new(ptr).map(|ptr| Self {
            ptr,
            layout,
            len,
            _marker: PhantomData,
        })
    }
}

impl<T> Drop for TrackedAllocation<T> {
    fn drop(&mut self) {
        // SAFETY: Layout matches what was used in allocation.
        // The pointer is valid because it came from a successful allocation.
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

// TrackedAllocation is Send if T is Send
unsafe impl<T: Send> Send for TrackedAllocation<T> {}

// TrackedAllocation is Sync if T is Sync
unsafe impl<T: Sync> Sync for TrackedAllocation<T> {}

/// Helper macro for documenting unsafe blocks
///
/// Use this pattern in your code:
///
/// ```rust,ignore
/// // SAFETY: This block is safe because:
/// // - Pointer `ptr` was validated via `validate_ptr()`
/// // - We have exclusive access to the memory
/// // - The length is checked to be within bounds
/// unsafe {
///     std::slice::from_raw_parts(ptr, len)
/// }
/// ```
#[macro_export]
macro_rules! safety_doc {
    ($($reason:tt)*) => {
        // SAFETY: $($reason)*
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_ptr_null() {
        let ptr: *const f32 = std::ptr::null();
        assert!(matches!(validate_ptr(ptr, 1), Err(SecurityError::NullPointer)));
    }

    #[test]
    fn test_validate_ptr_valid() {
        let data = vec![1.0f32, 2.0, 3.0];
        assert!(validate_ptr(data.as_ptr(), data.len()).is_ok());
    }

    #[test]
    fn test_validate_ptr_overflow() {
        let data = [0u8; 8];
        // Try to create a slice larger than max buffer
        assert!(matches!(
            validate_ptr(data.as_ptr(), MAX_FFI_BUFFER_SIZE + 1),
            Err(SecurityError::BufferTooLarge(_))
        ));
    }

    #[test]
    fn test_tracked_allocation() {
        let mut alloc = TrackedAllocation::<i32>::new(100).unwrap();
        assert_eq!(alloc.len(), 100);

        // Write data
        unsafe {
            for i in 0..100 {
                *alloc.as_mut_ptr().add(i) = i as i32;
            }

            // Verify data
            let slice = alloc.as_slice();
            for (i, &val) in slice.iter().enumerate() {
                assert_eq!(val, i as i32);
            }
        }

        // Allocation is automatically freed on drop
    }

    #[test]
    fn test_tracked_allocation_zeroed() {
        let alloc = TrackedAllocation::<u64>::new_zeroed(50).unwrap();

        unsafe {
            for &val in alloc.as_slice() {
                assert_eq!(val, 0);
            }
        }
    }

    #[test]
    fn test_tracked_allocation_copy_from_slice() {
        let mut alloc = TrackedAllocation::<f32>::new(4).unwrap();
        let source = [1.0f32, 2.0, 3.0, 4.0];

        alloc.copy_from_slice(&source);

        unsafe {
            assert_eq!(alloc.as_slice(), &source);
        }
    }

    #[test]
    fn test_tracked_allocation_into_raw_parts() {
        let alloc = TrackedAllocation::<u8>::new(64).unwrap();
        let (ptr, layout, len) = alloc.into_raw_parts();

        assert!(!ptr.is_null());
        assert_eq!(len, 64);

        // Reconstruct and drop properly
        unsafe {
            let _ = TrackedAllocation::<u8>::from_raw_parts(ptr, layout, len);
        }
    }

    #[test]
    fn test_slice_from_ptr() {
        let data = vec![1i16, 2, 3, 4, 5];

        unsafe {
            let slice = slice_from_ptr(data.as_ptr(), data.len()).unwrap();
            assert_eq!(slice, &[1, 2, 3, 4, 5]);
        }
    }

    #[test]
    fn test_slice_from_ptr_null() {
        let ptr: *const u32 = std::ptr::null();

        unsafe {
            assert!(matches!(
                slice_from_ptr(ptr, 10),
                Err(SecurityError::NullPointer)
            ));
        }
    }
}
