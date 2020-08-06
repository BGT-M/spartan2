from biosppy.signals import ecg

from .._model import DMmodel
from . import param_default, DTensor

param_default_dict = {
    'sampling_rate': 360,
    'left_size': 120,
    'right_size': 136,
    'out_path': None
}


class RPeak(DMmodel):
    def __init__(self, time_series, *args, **para_dict):
        self.series_data = time_series.val_tensor
        self.sampling_rate = param_default(para_dict, 'sampling_rate', param_default_dict)
        self.left_size = param_default(para_dict, 'left_size', param_default_dict)
        self.right_size = param_default(para_dict, 'right_size', param_default_dict)
        self.out_path = param_default(para_dict, 'out_path', param_default_dict)
        self.length = time_series.length
        self.segments = None

    def _find_rpeaks(self):
        sig_out = ecg.ecg(signal=self.series_data[0], sampling_rate=self.sampling_rate, show=False)
        r_peaks = sig_out["rpeaks"]
        segments = []
        for r_peak in r_peaks:
            if r_peak-self.left_size >= 0 and r_peak + self.right_size < self.length:
                ts = self.series_data[:, r_peak-self.left_size:r_peak + self.right_size]
                segments.append(ts)
        import numpy as np
        self.segments = DTensor.from_numpy(np.array(segments))
        if self.out_path is not None:
            np.save(self.out_path, segments)
        return self.segments

    def run(self):
        return self._find_rpeaks()
