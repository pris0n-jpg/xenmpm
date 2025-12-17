import numpy as np
import time
from dataclasses import dataclass, field


@dataclass
class Header:
    stamp: float = field(default_factory=lambda: time.time())
    frame_id: str = field(default_factory=str)




class Header:
    def __init__(self, stamp: float = None, frame_id: str = ""):
        self.stamp = stamp
        self.frame_id = frame_id
        if stamp is None:
            self.stamp = time.time()

    def __str__(self):
        return f"Header(stamp={self.stamp}, frame_id={self.frame_id}"

    def to_json(self) -> dict:
        return {"stamp": self.stamp, "frame_id": self.frame_id}

    def from_json(self, json):
        self.stamp = json["stamp"]
        self.frame_id = json["frame_id"]

     
class Message:
    """Empty base class for all messages."""

    def __init__(self, frame_id: str = ""):
        self.header = Header(frame_id=frame_id)
        
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        raise NotImplementedError
    
    def to_json(self) -> dict:
        raise NotImplementedError        
        

