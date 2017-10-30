import numpy as np
from typing import List, Set, Dict, Optional
from .interval import Interval
from .. import types
import logging

_logger = logging.getLogger(__name__)


class TargetsIntervalListMetadata:
    def __init__(self, targets_interval_list: List[Interval]):
        _logger.info("Generating targets metadata...")
        self.targets_interval_list = targets_interval_list
        self.num_targets = len(targets_interval_list)
        self.contig_set = self._get_contig_set_from_interval_list(targets_interval_list)
        self.contig_list = sorted(list(self.contig_set))
        self.num_contigs = len(self.contig_list)

        # map from contig to indices in the target list
        self.contig_target_indices: Dict[str, List[int]] = \
            {contig: [ti for ti in range(len(targets_interval_list))
                      if targets_interval_list[ti].contig == contig]
             for contig in self.contig_set}

        # number of targets per contig
        self.t_j = np.asarray([len(self.contig_target_indices[self.contig_list[j]])
                               for j in range(self.num_contigs)], dtype=types.big_uint)

    @staticmethod
    def _get_contig_set_from_interval_list(targets_interval_list: List[Interval]) -> Set[str]:
        return {target.contig for target in targets_interval_list}


class SampleCoverageMetadata:
    """ Represents essential metadata collected from a sample's coverage profile

    Note:
        The initializer only generates immediately available metadata (coverage per contig, etc.)
        The ploidy
    """
    def __init__(self,
                 sample_name: str,
                 n_t: np.ndarray,
                 targets_metadata: TargetsIntervalListMetadata):
        """
        :param sample_name: a string identifier
        :param n_t: read count array for the sample
        :param targets_metadata: targets interval list metadata
        """
        assert targets_metadata.num_targets == n_t.size
        self._targets_metadata = targets_metadata
        self.sample_name = sample_name

        # total count per contig
        self.n_j = np.zeros((targets_metadata.num_contigs,), dtype=types.big_uint)
        for j, contig in enumerate(targets_metadata.contig_list):
            target_indices = targets_metadata.contig_target_indices[contig]
            self.n_j[j] = np.sum(n_t[target_indices])

        # total count
        self.n_total = np.sum(self.n_j)

        # ploidy per contig (will be set by SampleCoverageMetadata.set_ploidy)
        self.ploidy_j: Optional[np.ndarray] = None

        # mean read depth per target per copy number (will be set by SampleCoverageMetadata.set_ploidy)
        self.read_depth: Optional[float] = None

    def set_ploidy_and_read_depth(self, ploidy_j: np.ndarray):
        """
        :param ploidy_j: a vector of ploidy per contig
        :return:
        """
        assert self._targets_metadata.num_contigs == ploidy_j.size
        self.ploidy_j = np.zeros((self._targets_metadata.num_contigs,), dtype=types.small_uint)
        self.ploidy_j[:] = ploidy_j[:]
        self.read_depth = float(self.n_total) / np.sum(self.ploidy_j * self._targets_metadata.t_j)

    @property
    def has_ploidy_metadata(self):
        return self.ploidy_j is not None

    @property
    def has_read_depth_metadata(self):
        return self.read_depth is not None


class SampleCoverageMetadataCollection:
    def __init__(self,
                 sample_names: List[str],
                 n_st: np.ndarray,
                 targets_metadata: TargetsIntervalListMetadata):
        _logger.info("Generating sample coverage metadata...")
        assert len(set(sample_names)) == len(sample_names), "samples in the collection must have unique names"
        self.sample_names = sample_names
        self.num_samples = len(sample_names)
        self.sample_metadata_list = [SampleCoverageMetadata(sample_name, n_st[si, :], targets_metadata)
                                     for si, sample_name in enumerate(sample_names)]

        self._sample_name_to_sample_index = {sample_name: si for si, sample_name in enumerate(sample_names)}
        self._sample_names_set = set(sample_names)

    @property
    def all_have_ploidy_and_read_depth_metadata(self):
        return all([sample_metadata.has_ploidy_metadata and sample_metadata.has_read_depth_metadata
                    for sample_metadata in self.sample_metadata_list])

    def get_sample_coverage_metadata_by_name(self, sample_name: str):
        assert sample_name in self._sample_names_set,\
            'a sample named "{0}" is not in the collection'.format(sample_name)
        return self.sample_metadata_list[self._sample_name_to_sample_index[sample_name]]

    def get_sample_coverage_metadata_by_index(self, sample_index: int):
        assert 0 <= sample_index <= self.num_samples, "sample index out of range"
        return self.sample_metadata_list[sample_index]

