from .std_msgs import Header, dataclass, field
import numpy as np


@dataclass
class XenseMessage:
    header: Header = field(default_factory=Header)
    Rectify: np.ndarray = None
    Difference: np.ndarray = None
    Depth: np.ndarray = None
    Marker2D: np.ndarray = None
    Force: np.ndarray = None

    def __str__(self):
        msg = "Xense Message: \n"
        msg += str(self.header)
        msg += f"Rectify shape: {None if self.Rectify is None else self.Rectify.shape}\n"
        msg += f"Difference shape: {None if self.Difference is None else self.Difference.shape}\n"
        msg += f"Depth shape: {None if self.Depth is None else self.Depth.shape}\n"
        msg += f"Marker2D shape: {None if self.Marker2D is None else self.Marker2D.shape}\n"
        msg += f"Force shape: {None if self.Force is None else self.Force.shape}\n"
        return msg