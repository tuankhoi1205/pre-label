from dataclasses import asdict, dataclass, replace
import numpy as np


@dataclass(frozen=True)
class Cuboid:
    sample_token: str
    frame_id: int
    prediction_id: str
    class_name: str
    confidence: float | None
    center_xyz: list[float]
    size_lwh: list[float]
    quaternion_wxyz: list[float]
    coordinate_frame: str = "lidar_sensor"
    model: str = ""
    checkpoint: str = ""
    annotation_id: int | None = None
    identity_source: str = "prediction"

    def __post_init__(self):
        for key, length in [("center_xyz", 3), ("size_lwh", 3), ("quaternion_wxyz", 4)]:
            a = np.asarray(getattr(self, key), dtype=float)
            if a.shape != (length,) or not np.isfinite(a).all():
                raise ValueError(f"Invalid {key}")
        if min(self.size_lwh) <= 0:
            raise ValueError("Box dimensions must be strictly positive")
        if not np.isclose(np.linalg.norm(self.quaternion_wxyz), 1, atol=1e-5):
            raise ValueError("Quaternion must be unit length, scalar first")
        if self.confidence is not None and (not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError("Confidence must be in [0,1] or null")
        if not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def updated(self, **kwargs):
        return replace(self, **kwargs)


def ensure_same_frame(a, b):
    if (a.sample_token, a.coordinate_frame) != (b.sample_token, b.coordinate_frame):
        raise ValueError("Geometry comparison requires the same sample and coordinate frame")
