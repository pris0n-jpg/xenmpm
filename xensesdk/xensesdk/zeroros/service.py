import logging
import threading
from typing import Callable
from .utils.req_rep import ReqRepClient, ReqRepServer



class Service:
    
    def __init__(self, port: int=5556, timeout_ms: int=800):
        self.port = port
        self._callback_dict = {}
        
        self.server = ReqRepServer(
            port=port, 
            impl_callback=self.impl_callback,
            timeout_ms=timeout_ms,
            log_level=logging.WARNING, 
            is_encrypt=True
        )
    
    def impl_callback(self, payload: dict):
        """
        Implementation of the callback function to handle incoming messages.
        :param payload: The incoming payload from the client.
        """
        action_key = payload.get("action_key")
        if action_key in self._callback_dict:
            callback = self._callback_dict[action_key]
            try:
                return {
                    "success": True,
                    "ret": callback(*payload["args"], **payload["kwargs"])
                }
            except:
                return {"success": False, "message": f"Error in: {action_key}"}
        elif action_key == "echo":
            payload["success"] = True
            return payload
        else:
            return {"success": False, "message": f"No action: {action_key}"}
        
    def start(self, threaded: bool = True):
        """
        Starts the server, defaulting to blocking mode
            :param threaded: Whether to start the server in a separate thread
        """
        if threaded:
            self.thread = threading.Thread(target=self.server.run, daemon=True)
            self.thread.start()
        else:
            self.server.run()
    
    def stop(self):
        """
        Stops the server
        """
        self.server.stop()
    
    def reset(self):
        self.server.reset()
    
    def register_callback(self, action_key: str, callback: Callable):
        """
        Registers a callback function for a specific action key.
            :param action_key: The action key to register the callback for
            :param callback: The callback function to register
        """
        self._callback_dict[action_key] = callback

    
    
class ServiceProxy:
    
    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 5556,
    ):
        """
        Initializes the ServiceProxy.
            :param ip: The IP address of the server
            :param port: The port of the server
            :param wait_for_server:  Whether to retry connecting to the server
        """
        self.client = ReqRepClient(
            ip=ip, 
            port=port,
            timeout_ms=800,
            log_level=logging.WARNING,
            is_encrypt=True, 
        )
    
    def request(self, action_key: str, args: list = None, kwargs: dict = None):
        """
        Sends a request to the server.
            :param action_key: The action key to send
            :param args: The arguments to send with the request
        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        ret = self.client.send_msg({"action_key": action_key, "args": args, "kwargs": kwargs})
        if not ret:
            ret = {"success": False, "message": "Request timeout."}
        return ret
    
    def __getattr__(self, action_key: str):
        """
        Some magic allows dynamic access to the request method.
            :param action_key: The action key to send
        """
        def wrapper(*args, **kwargs):
            return self.request(action_key, list(args), kwargs)
        return wrapper
    
    def is_connected(self) -> bool:
        """
        Tests the connection to the server.
        """
        ret = self.request("echo")
        return ret["success"]
    
    def stop(self):
        """
        Stops the client.
        """
        self.client.stop()
    
    def reset(self):
        self.client.reset()