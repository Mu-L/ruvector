//! Epoch-based reconciliation for hybrid RVF + IndexedDB persistence.
//!
//! RVF is the source of truth for vectors. IndexedDB is a rebuildable
//! cache for metadata. Both stores share a monotonic epoch counter.
//!
//! Write order:
//! 1. Write vectors to RVF (append-only, crash-safe)
//! 2. Write metadata to IndexedDB
//! 3. Commit shared epoch in both stores
//!
//! On startup: compare epochs and rebuild the lagging side.

/// Monotonic epoch counter shared between RVF and metadata stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Epoch(pub u64);

impl Epoch {
    pub const ZERO: Self = Self(0);

    pub fn next(self) -> Self {
        Self(self.0.checked_add(1).expect("epoch overflow"))
    }

    pub fn value(self) -> u64 {
        self.0
    }
}

/// Result of comparing epochs between RVF and metadata stores.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconciliationAction {
    /// Both stores are in sync -- no action needed.
    InSync,
    /// RVF is ahead -- rebuild metadata from RVF vectors.
    RebuildMetadata { rvf_epoch: Epoch, metadata_epoch: Epoch },
    /// Metadata is ahead (should not happen) -- log warning, trust RVF.
    TrustRvf { rvf_epoch: Epoch, metadata_epoch: Epoch },
}

/// Compare epochs and determine reconciliation action.
pub fn reconcile(rvf_epoch: Epoch, metadata_epoch: Epoch) -> ReconciliationAction {
    match rvf_epoch.cmp(&metadata_epoch) {
        std::cmp::Ordering::Equal => ReconciliationAction::InSync,
        std::cmp::Ordering::Greater => ReconciliationAction::RebuildMetadata {
            rvf_epoch,
            metadata_epoch,
        },
        std::cmp::Ordering::Less => ReconciliationAction::TrustRvf {
            rvf_epoch,
            metadata_epoch,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_sync() {
        let e = Epoch(5);
        assert_eq!(reconcile(e, e), ReconciliationAction::InSync);
    }

    #[test]
    fn rvf_ahead_rebuilds_metadata() {
        let action = reconcile(Epoch(3), Epoch(2));
        assert_eq!(
            action,
            ReconciliationAction::RebuildMetadata {
                rvf_epoch: Epoch(3),
                metadata_epoch: Epoch(2),
            }
        );
    }

    #[test]
    fn metadata_ahead_trusts_rvf() {
        let action = reconcile(Epoch(1), Epoch(3));
        assert_eq!(
            action,
            ReconciliationAction::TrustRvf {
                rvf_epoch: Epoch(1),
                metadata_epoch: Epoch(3),
            }
        );
    }

    #[test]
    fn epoch_increment() {
        assert_eq!(Epoch::ZERO.next(), Epoch(1));
        assert_eq!(Epoch(99).next(), Epoch(100));
    }
}
