
import os
import zlib
import lz4.frame
import pickle
import time
from typing import Tuple, Callable, Any, Union

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

def validate_topic(topic):
    # Verify that the topic has no spaces
    if " " in topic:
        raise ValueError("Topic name cannot contain spaces")

    # Verify that the topic starts with a slash
    if not topic.startswith("/"):
        raise ValueError("Topic name must start with a slash")

    # Verify that the topic does not end with a slash
    if topic.endswith("/"):
        raise ValueError("Topic name must not end with a slash")
    return topic

# from agentlace
def make_compression_method(compression: str) -> Tuple[Callable, Callable]:
    """
    NOTE: lz4 is faster than zlib, but zlib has better compression ratio
        :return: compress, decompress functions
            def compress(object) -> bytes
            def decompress(data) -> object
    TODO: support msgpack
    """
    if compression == 'lz4':
        def compress(data): return lz4.frame.compress(pickle.dumps(data, protocol=4))
        def decompress(data): 
            data = lz4.frame.decompress(data)
            data = pickle.loads(data)
            return data
            # return pickle.loads(lz4.frame.decompress(data))
    elif compression == 'zlib':
        def compress(data): return zlib.compress(pickle.dumps(data, protocol=4))
        def decompress(data): return pickle.loads(zlib.decompress(data))
    else:
        raise Exception(f"Unknown compression algorithm: {compression}")
    return compress, decompress



def encrypt(bytes_data: bytes, password) -> bytes:
    # 生成密钥
    salt = os.urandom(16)  # 盐值
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=default_backend())
    key = kdf.derive(password.encode())  # 从密码派生密钥
    # 生成随机的初始化向量 (IV)
    iv = os.urandom(12)
    # 设置 AES-GCM 加密
    encryptor = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend()).encryptor()

    ciphertext = encryptor.update(bytes_data) + encryptor.finalize()
    return salt + iv + encryptor.tag + ciphertext

def decrypt(bytes_data: bytes, password) -> bytes:
    salt = bytes_data[:16]  # 盐值
    iv = bytes_data[16:28]  # 初始化向量 (IV)
    tag = bytes_data[28:44]  # GCM TAG
    ciphertext = bytes_data[44:]  # 密文
    # 从密码派生密钥
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=default_backend())
    key = kdf.derive(password.encode())
    # 设置 AES-GCM 解密
    decryptor = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend()).decryptor()

    # 解密模型
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    return plaintext


if os.name == 'nt':
    import ctypes
    from ctypes.wintypes import LARGE_INTEGER

    kernel32 = ctypes.windll.kernel32
    INFINITE = 0xFFFFFFFF
    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
    def accurate_sleep(seconds):
        """
        accurate sleep function for Windows.

        Parameters:
        - seconds : float
        """
        handle = kernel32.CreateWaitableTimerExW(
            None, None, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, 0x1F0003
        )
        res = kernel32.SetWaitableTimer(
            handle,
            ctypes.byref(LARGE_INTEGER(int(seconds * -10000000))),
            0, None, None, 0,
        )
        res = kernel32.WaitForSingleObject(handle, INFINITE)
        kernel32.CancelWaitableTimer(handle)
    
else:
    def accurate_sleep(seconds):
        time.sleep(seconds)
        
