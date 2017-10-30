import numpy as np
import pymc3 as pm
from typing import Callable


class ParamTrackerConfig:
    def __init__(self):
        self.param_names = []
        self.inv_trans_list = []
        self.inv_trans_param_names = []

    def add(self,
            param_name: str,
            inv_trans: Callable[[np.ndarray], np.ndarray],
            inv_trans_param_name: str):
        self.param_names.append(param_name)
        self.inv_trans_list.append(inv_trans)
        self.inv_trans_param_names.append(inv_trans_param_name)


class ParamTracker:
    def __init__(self, param_tracker_config: ParamTrackerConfig):
        self.param_tracker_config = param_tracker_config
        self.tracked_param_values_dict = {}
        for key in self.param_tracker_config.inv_trans_param_names:
            self.tracked_param_values_dict[key] = []

    # todo there must a way of doing this without accessing approx._global_view
    def _extract_param_mean(self, approx: pm.approximations.MeanField):
        all_means = approx.mean.eval()
        out = dict()
        for param_name, inv_trans, inv_trans_param_name in zip(
                self.param_tracker_config.param_names,
                self.param_tracker_config.inv_trans_list,
                self.param_tracker_config.inv_trans_param_names):
            _, slc, _, dtype = approx._global_view[param_name]
            bare_param_mean = all_means[..., slc].astype(dtype)
            if inv_trans is None:
                out[inv_trans_param_name] = bare_param_mean
            else:
                out[inv_trans_param_name] = inv_trans(bare_param_mean)
        return out

    def record(self, approx, _loss, _i):
        out = self._extract_param_mean(approx)
        for key in self.param_tracker_config.inv_trans_param_names:
            self.tracked_param_values_dict[key].append(out[key])

    __call__ = record

    def clear(self):
        for key in self.param_tracker_config.inv_trans_param_names:
            self.tracked_param_values_dict[key] = []

    def __getitem__(self, key):
        return self.tracked_param_values_dict[key]
