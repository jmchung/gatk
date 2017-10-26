import numpy as np
from ..utils.interval import Interval
from typing import List, Set, Tuple, Optional
import logging

__all__ = ['TargetMask']

_logger = logging.getLogger(__name__)


class TargetMask:
    def __init__(self, num_targets: int):
        self.num_targets = num_targets
        self.drop_t = np.zeros((num_targets,), dtype=bool)
        self.drop_reason_t: List[Set[str]] = [set() for _ in range(num_targets)]

    def _assert_mask_compatibility_with_target_interval_list(self, targets_interval_list: List[Interval]):
        assert len(targets_interval_list) == self.num_targets, \
            "Mask number of targets ({0}) is not compatible with the provided " \
            "targets interval list (length = {1})".format(self.num_targets, len(targets_interval_list))

    def _assert_mask_compatibility_with_read_count_array(self, n_st: np.ndarray):
        assert n_st.shape[1] == self.num_targets, \
            "Mask number of targets ({0}) is not compatible with the provided " \
            "read count array (shape = {1})".format(self.num_targets, n_st.shape)

    def _assert_mask_compatibility(self, targets_interval_list: Optional[List[Interval]], n_st: Optional[np.ndarray]):
        if targets_interval_list is not None:
            self._assert_mask_compatibility_with_target_interval_list(targets_interval_list)
        if n_st is not None:
            self._assert_mask_compatibility_with_read_count_array(n_st)

    @staticmethod
    def _assert_read_count_int_dtype(n_st: np.ndarray):
        assert n_st.dtype in [np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64], \
            "can not reliably detect cohort-wide uncovered targets with the dtype of the given " \
            "read counts array ({0})".format(n_st.dtype)

    def get_masked_view(self, targets_interval_list: List[Interval], n_st: np.ndarray):
        """
        Applies the mask on a given targets interval list and read count array
        :return: (a view of the provided n_st,
                  a new list containing references to the provided targets interval list)
        """
        self._assert_mask_compatibility(targets_interval_list, n_st)
        kept_targets_indices = [ti for ti in range(len(targets_interval_list)) if not self.drop_t[ti]]
        num_dropped_targets = self.num_targets - len(kept_targets_indices)
        kept_targets_interval_list = [targets_interval_list[ti] for ti in kept_targets_indices]
        kept_n_st = n_st[:, kept_targets_indices]
        if num_dropped_targets > 0:
            dropped_fraction = num_dropped_targets / self.num_targets
            _logger.warning("Some targets have been dropped. Dropped fraction: {0:2.6}".format(dropped_fraction))
        return kept_n_st, kept_targets_interval_list

    def keep_only_given_contigs(self, contigs_to_keep: Set[str], targets_interval_list: List[Interval]):
        self._assert_mask_compatibility(targets_interval_list, None)
        inactive_target_indices = [target.contig not in contigs_to_keep for target in targets_interval_list]
        self.drop_t[inactive_target_indices] = True
        for ti in inactive_target_indices:
            self.drop_reason_t[ti].add("contig marked to be dropped")

    def drop_cohort_wide_uncovered_targets(self, n_st: np.ndarray):
        self._assert_mask_compatibility(None, n_st)
        self._assert_read_count_int_dtype(n_st)
        for ti in range(self.num_targets):
            if all(n_st[:, ti] == 0):
                self.drop_t[ti] = True
                self.drop_reason_t[ti].add("cohort-wide uncovered target")

    def drop_targets_with_anomalous_coverage(self):
        raise NotImplementedError

