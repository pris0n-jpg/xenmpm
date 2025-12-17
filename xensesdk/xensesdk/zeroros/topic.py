import zmq
from threading import Lock, Thread

from .messages import Message
from .utils.general_utils import make_compression_method, validate_topic, accurate_sleep
import time

class Publisher:
    def __init__(
        self, 
        topic: str, 
        message_class = object, 
        port: int = 5555, 
        compression: str = 'lz4', 
        hwm: int = 20
    ):
        self.topic = topic.encode('utf-8')
        self.hwm = hwm
        self.port = port
        self.message_class = message_class
        self.compress, _ = make_compression_method(compression)
        
        self._lock = Lock()  # 线程安全锁
        self.context = None
        self.sock = None
        self.reset()  # 初始化套接字连接

    def publish(self, message: Message):
        if not isinstance(message, self.message_class):
            raise TypeError("Message type mismatch")
        
        with self._lock:
            # 多帧发送：主题帧 + 数据帧
            msg_bytes = self.compress(message)
            self.sock.send_multipart([self.topic, msg_bytes], copy=False)

    def reset(self):        
        """重置套接字连接"""
        self.context = zmq.Context()
        if self.sock:
            self.sock.close()
            self.sock = None
        self.sock = self.context.socket(zmq.PUB)
        self.sock.setsockopt(zmq.LINGER, 0)  # 避免关闭阻塞
        self.sock.setsockopt(zmq.SNDHWM,self. hwm)  # 发送队列水位控制
        self.sock.bind(f"tcp://*:{self.port}")

    def stop(self):
        if self.sock:
            self.sock.close()
        self.context.term()
        self.socket = None
        self.context = None
    
    def test_delay(self):
        while True:
            time.sleep(1)
            self.publish(Message())



class Subscriber:
    def __init__(
        self, 
        topic: str, 
        message_class = object,
        callback: callable = None,
        ip: str = "127.0.0.1", 
        port: int = 5555, 
        compression: str = 'lz4', 
        hwm: int = 20
    ):
        self.topic = topic.encode('utf-8')
        self.message_class = message_class
        self.url = f"tcp://{ip}:{port}"
        self.callback = callback
        self.hwm = hwm
        self.is_stop = False
        _, self.decompress = make_compression_method(compression)
        # 连接初始化
        self.context = None
        self.sock = None
        self.reset()
        
        # 启动独立接收线程
        self.recv_thread = Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()

    def _recv_loop(self):
        while not self.is_stop:
            try:
                frames = self.sock.recv_multipart(flags=zmq.NOBLOCK)
                if len(frames) == 2:
                    recv_topic, data = frames
                    if self.callback:
                        msg = self.decompress(data)
                        if not isinstance(msg, self.message_class):
                            raise TypeError("Message type mismatch")
                        # 调用回调函数处理消息
                        self.callback(msg)

            except zmq.Again:
                accurate_sleep(0.002)  # 避免空转消耗CPU, win 必须使用 accurate_sleep

            except zmq.ZMQError as e:
                print(f"Connection error: {e}, reconnecting...")
                try:
                    self.reset()
                except Exception as e:
                    print(f"Reconnect failed: {e}")
                    break

    def reset(self):
        """断线重连机制"""
        self.context = zmq.Context()
        if self.sock:
            self.sock.close()
        self.sock = self.context.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, self.topic)
        self.sock.setsockopt(zmq.RCVHWM, self.hwm)
        self.sock.connect(self.url)
        self.is_stop = False

    def stop(self):
        self.is_stop = True
        if self.sock:
            self.sock.close()
        self.context.term()
        self.context = None
        self.socket = None
    
    def test_delay(self):
        def callback(msg):
            if hasattr(msg, 'header'):
                print(f"Delay: {(time.time()-msg.header.stamp) * 1000:.3f} ms")
        self.callback = callback
        time.sleep(30)