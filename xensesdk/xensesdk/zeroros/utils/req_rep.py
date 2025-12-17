import logging
from threading import Lock
from typing import Optional, Dict, Protocol, Union
import zmq
import traceback

from .general_utils import make_compression_method, decrypt, encrypt


class CallbackProtocol(Protocol):
    def __call__(self, message: Dict) -> Dict:
        ...


class ReqRepServer:
    def __init__(
        self,
        port=5556,
        impl_callback: Optional[CallbackProtocol] = None,
        timeout_ms=800,
        log_level=logging.DEBUG,
        compression: str = 'lz4',
        is_encrypt: bool = False,
    ):
        """
        Request reply server
        """
        self.impl_callback = impl_callback
        self.timeout_ms = timeout_ms
        self.port = port
        self.is_encrypt = is_encrypt
        self.is_stop: bool = None
        self.compress, self.decompress = make_compression_method(compression)
        self.socket = None
        self.context = None
        self.reset()
        logging.basicConfig(level=log_level)
        logging.debug(f"Req-rep server is listening on port {port}")

    def run(self):
        while not self.is_stop:
            try:
                #  Wait for next request from client
                message = self.socket.recv()
                if self.is_encrypt:
                    password = "Wz8mmWz2ALJ6X5Ic"
                    message = decrypt(message, password)
                message = self.decompress(message)
                logging.debug(f"Received new request: {message}")

                #  Send reply back to client
                if self.impl_callback:
                    ret = self.impl_callback(message)
                    message = self.compress(ret)
                    if self.is_encrypt:
                        password = "Wz8mmWz2ALJ6X5Ic"
                        message = encrypt(message, password)
                    self.socket.send(message)
                else:
                    logging.warning("No implementation callback provided.")
                    self.socket.send(b"World")
            except zmq.Again as e:
                continue
            except zmq.ZMQError as e:
                # Handle ZMQ errors gracefully
                if self.is_stop:
                    logging.debug("Stopping the ZMQ server...")
                    break
                else:
                    raise e

    def stop(self):
        self.is_stop = True
        if self.socket:
            self.socket.close()
        self.context.term()
        self.context = None
        self.socket = None

    def reset(self):
        self.context = zmq.Context()
        if self.socket:
            self.socket.close()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.SNDHWM, 5)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{self.port}")
        self.is_stop = False


##############################################################################

class ReqRepClient:
    def __init__(
        self,
        ip: str,
        port=5556,
        timeout_ms=800,
        log_level=logging.DEBUG,
        compression: str = 'lz4',
        is_encrypt: bool = False,
    ):
        """
        :param ip: IP address of the server
        :param port: Port number of the server
        :param timeout_ms: Timeout in milliseconds
        :param log_level: Logging level, defaults to DEBUG
        :param compression: Compression algorithm, defaults to lz4
        """
        logging.basicConfig(level=log_level)
        logging.debug(f"Req-rep client is connecting to {ip}:{port}")

        self.compress, self.decompress = make_compression_method(compression)
        self.is_encrypt = is_encrypt
        self.ip, self.port, self.timeout_ms = ip, port, timeout_ms
        self._lock = Lock()
        self.socket = None
        self.context = None
        self.reset()

    def reset(self):
        """
        Reset the socket connection, this is needed when REQ is in a
        broken state.
        """
        self.context = zmq.Context()
        if self.socket:
            self.socket.close()
            
        try:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close
            self.socket.connect(f"tcp://{self.ip}:{self.port}")
        except Exception as e:
            logging.error(f"Failed to create new socket: {e}")
            self.socket = None
            raise

    def send_msg(self, request: dict, wait_for_response=True) -> Union[None, str, dict]:
        if self.socket is None:
            logging.warning("Socket is None, attempting reset...")
            try:
                self.reset()
            except Exception as e:
                logging.error(f"Failed to reset socket: {e}")
                return None

        with self._lock:
            try:
                serialized = self.compress(request)
                if self.is_encrypt:
                    password = "Wz8mmWz2ALJ6X5Ic"
                    serialized = encrypt(serialized, password)
                self.socket.send(serialized)
                if wait_for_response is False:
                    return None
                message = self.socket.recv()
                if self.is_encrypt:
                    password = "Wz8mmWz2ALJ6X5Ic"
                    message = decrypt(message, password)
                return self.decompress(message)
            except Exception as e:
                logging.warning(
                    f"Failed to send message to {self.ip}:{self.port}")
                logging.warning(f"{e}: {traceback.format_exc()}")
                try:
                    self.reset()
                except Exception as reset_error:
                    logging.error(f"Failed to reset socket after error: {reset_error}")
                return None

    def stop(self):
        if self.socket:
            self.socket.close()
        self.context.term()
        self.context = None
        self.socket = None
