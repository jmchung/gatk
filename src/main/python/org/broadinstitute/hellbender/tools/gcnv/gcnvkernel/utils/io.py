import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Tuple

from .interval import Interval, IntervalAnnotation, interval_annotations_dict, interval_annotations_dtypes
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# standard read counts and target interval list files data types
std_dtypes_dict = {'contig': np.str, 'start': np.int, 'stop': np.int, 'name': np.str}


def _get_tsv_header(tsv_filename: str, comment: str = '#') -> str:
    """ Extracts the header line from a .tsv file. The header line is the first non-commented line.
    :param tsv_filename: (string) tsv file
    :param comment: (string) comment character
    :return: string
    """
    with open(tsv_filename, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            if len(stripped_line) == 0 or stripped_line[0] == comment:
                continue
            return stripped_line
        raise Exception("Header line could not be found")


def _convert_targets_pd_to_interval_list(targets_pd: pd.DataFrame) -> List[Interval]:
    """
    Converts a pandas dataframe targets intervals to list(Interval). Annotations will be parsed
    and added to the intervals as well.
    """
    interval_list: List[Interval] = []
    columns = [str(x) for x in targets_pd.columns.values]
    assert all([required_column in columns for required_column in std_dtypes_dict.keys()]), "Some columns missing"
    for contig, start, stop, name in zip(targets_pd['contig'], targets_pd['start'],
                                         targets_pd['stop'], targets_pd['name']):
        interval = Interval(contig, start, stop, name)
        interval_list.append(interval)

    for annotation_key in set(columns).intersection(interval_annotations_dict.keys()):
        bad_annotations_found = False
        for ti, raw_value in enumerate(targets_pd[annotation_key]):
            try:
                annotation: IntervalAnnotation = interval_annotations_dict[annotation_key](raw_value)
                interval_list[ti].add_annotation(annotation_key, annotation)
            except ValueError:
                bad_annotations_found = True
        if bad_annotations_found:
            _logger.warning("Some of the annotations for {0} contained bad values and were ignored".format(
                annotation_key))

    return interval_list


def load_read_counts_tsv_file(read_counts_tsv_file: str, read_counts_data_type=np.int, max_rows: Optional[int] = None)\
        -> Tuple[np.ndarray, List[str], List[Interval]]:
    header = _get_tsv_header(read_counts_tsv_file)
    sample_names = header.split()[4:]
    assert len(set(sample_names)) == len(sample_names), "Sample names are not unique."

    count_dtype_dict = dict()
    for sample_name in sample_names:
        count_dtype_dict[sample_name] = read_counts_data_type
    counts_pd = pd.read_csv(read_counts_tsv_file, delimiter='\t', nrows=max_rows,
                            dtype={**std_dtypes_dict, **count_dtype_dict})

    targets_pd = counts_pd[list(std_dtypes_dict.keys())]
    targets_interval_list = _convert_targets_pd_to_interval_list(targets_pd)
    counts_array_st: np.ndarray = counts_pd.loc[:, sample_names].as_matrix().T

    return counts_array_st, sample_names, targets_interval_list


def load_targets_tsv_file(targets_tsv_file: str) -> List[Interval]:
    targets_pd = pd.read_csv(targets_tsv_file, delimiter='\t',
                             dtype={**std_dtypes_dict, **interval_annotations_dtypes})
    return _convert_targets_pd_to_interval_list(targets_pd)


def load_data(read_counts_tsv_file: str, targets_tsv_file: Optional[str], **kwargs)\
        -> Tuple[np.ndarray, List[str], List[Interval]]:
    """ Loads read count data and optionally a (annotated) targets file.
    :param read_counts_tsv_file:
    :param targets_tsv_file:
    :param kwargs: (see load_read_counts_tsv_file)
    :return: a tuple of counts, sample names, and a list of intervals
    """
    _logger.info("Loading read counts file...")
    counts_array_st, sample_names, counts_targets_interval_list = load_read_counts_tsv_file(read_counts_tsv_file,
                                                                                            **kwargs)
    out_targets_interval_list = counts_targets_interval_list
    if targets_tsv_file is not None:
        _logger.info("Targets file provided; loading targets file...")
        loaded_targets_interval_list = load_targets_tsv_file(targets_tsv_file)
        counts_targets_interval_set = set(counts_targets_interval_list)
        mutual_targets_interval_list = [interval for interval in loaded_targets_interval_list
                                        if interval in counts_targets_interval_set]
        assert len(mutual_targets_interval_list) == len(counts_targets_interval_list),\
            "Some of the targets in the read counts file are absent in the targets file"
        out_targets_interval_list = mutual_targets_interval_list

    return counts_array_st, sample_names, out_targets_interval_list
