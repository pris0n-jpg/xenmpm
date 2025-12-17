import numpy as np
from xensesdk.omni import transforms as tf
from xensesdk.ezgl.functions import CircularBuffer
from xensesdk.zeroros import Node
from threading import Event
from xensesdk.xenseInterface.sensorEnum import OutputType

class VirtualRectify(tf.Rectify):
    def __call__(self, img):
        return img

class DataNode(Node):
    
    def __init__(
        self,
        ip_address,
        camera_serial,
    ):
        assert isinstance(ip_address, str), "Device address must be a string: xxx.xxx.xxx.xxx"
        assert isinstance(camera_serial, str), "Camera serial number must be a string"
        super().__init__(ip_address, name=camera_serial)
        
        self.buffer = CircularBuffer(2)
        self.client = self.create_client(camera_serial)
        self.subscriber = self.create_subscriber(camera_serial, self.on_fetch_data)
        self.new_data_event = Event()
    
    def reset_fetch_types(self, types):
        self.client.reset_fetch_types(types)       
        self.buffer.clear()
        self.new_data_event.clear()
        
    def on_fetch_data(self, msg):
        self.new_data_event.set()
        return self.buffer.put(msg)
    
    def get_data(self, timeout=0):
        if self.new_data_event.wait(timeout):
            return self.buffer.get()
        else:
            return None
    
class RemoteCamera:
    
    def __init__(
        self, 
        camera_serial, 
        ip_address, 
        size=(640, 480), 
        **kwargs
    ):
        self.rectify = VirtualRectify(2, 2, size)
        self.serial_number = camera_serial
        self.device_ip = ip_address
        self.types_from_remote = set()
        
        self.data_node = None
        self.init(camera_serial)
        self.data_node.reset_fetch_types(self.types_from_remote)

    def init(self, serial_number):
        self.release()
        self.serial_number = serial_number
        self.data_node = DataNode(self.device_ip, self.serial_number)
    
    def reset_reference(self):
        if self.data_node is not None:
            self.data_node.client.reset_reference()

    def get_data(self, types: list):        
        if set(types).issubset(self.types_from_remote):
            return self.data_node.get_data()
        
        else: # reset remote camera
            self.types_from_remote |= set(types)
            self.data_node.reset_fetch_types(self.types_from_remote)
            return self.data_node.get_data(2)

    def get_frame(self):
        return self.get_raw_frame()

    def get_raw_frame(self):       
        data_cache = self.get_data([OutputType.Rectify])
        if data_cache is not None:
            return True, data_cache[OutputType.Rectify]
        else:
            return False, None
    
    def set_diff_enabled(self, b):
        pass
    
    def destroyFetchThread(self):
        self.release()
    
    def release(self):
        if self.data_node is not None:
            self.data_node.destroy_node()
            self.data_node = None
