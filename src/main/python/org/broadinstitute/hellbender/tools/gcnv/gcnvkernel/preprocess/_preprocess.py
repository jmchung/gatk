import numpy as np
from ..utils.interval import Interval
from typing import List, Set, Tuple


__all__ = ['get_default_target_mask',
           'update_mask_kept_contigs',
           'apply_mask']


def _assert_mask_compatible_with_target_interval_list(targets_interval_list: List[Interval],
                                                      mask_t: np.ndarray):
    assert len(targets_interval_list) == len(mask_t),\
        "Mask shape ({0}) is not compatible with the provided targets interval list (length = {1})".format(
        len(mask_t), len(targets_interval_list))


def get_default_target_mask(num_targets: int) -> np.ndarray:
    """
    Returns an all-inclusive target mask
    :return: a boolean ndarray
    """
    return np.zeros((num_targets,), dtype=bool)


def update_mask_kept_contigs(contigs_to_keep: Set[str],
                             targets_interval_list: List[Interval],
                             mask_t: np.ndarray) -> None:
    _assert_mask_compatible_with_target_interval_list(targets_interval_list, mask_t)
    inactive_target_indices = [target.contig not in contigs_to_keep for target in targets_interval_list]
    mask_t[inactive_target_indices] = True


def apply_mask(targets_interval_list: List[Interval],
               n_st: np.ndarray,
               mask_t: np.ndarray) -> Tuple[np.ndarray, List[Interval]]:
    kept_targets_indices = [ti for ti in range(len(targets_interval_list)) if not mask_t[ti]]
    kept_targets_interval_list = [targets_interval_list[ti] for ti in kept_targets_indices]
    kept_n_st = n_st[:, kept_targets_indices]
    return kept_n_st, kept_targets_interval_list

