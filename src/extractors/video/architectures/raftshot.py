import numpy as np

from ....utils.config import Config
from ..zoom_features import ZoomFeatureExtractor
from .common import robust_minmax


def raft_boundary_signal(video_path: str, config: Config) -> np.ndarray:
    flow = ZoomFeatureExtractor(video_path=video_path, config=config).run()
    cols = [col for col in ("flow_mag_med", "radial_med", "radial_ratio") if col in flow.columns]
    if not cols:
        return np.zeros(len(flow), dtype=np.float32)
    values = flow[cols].to_numpy(dtype=np.float32)
    return robust_minmax(np.r_[0.0, np.linalg.norm(np.diff(values, axis=0), axis=1)])
