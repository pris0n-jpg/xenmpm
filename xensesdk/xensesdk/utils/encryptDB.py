"""
Description: 加密数据字典

Author: Jin Liu
Date: 2025/04/08
"""


from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import io
import os
import zlib
import pickle


class EncryptDB:
    """
    EncryptDB 类用于加密和解密数据文件, 后缀为 .edb
    """

    def __init__(self, compress: bool=False, input_path: str=None, password: str=None):
        self._compress = compress
        self._data = {}
        if input_path:
            self.load(input_path, password)
    
    def keys(self):
        return self._data.keys()
    
    def pop(self, key: str):
        """
        从字典中删除指定的键
        """
        if key not in self._data:
            print(f"{key} not found in data")
        
        return self._data.pop(key)
    
    def add_data(self, key:str, data):
        """
        添加常见数据类型到字典中
        """
        self._data[key] = data

    def add_file(self, key: str, file_path):
        """add .pth, .onnx, .yaml, etc."""
        
        with open(file_path, "rb") as f:
            plaintext = f.read()

        self._data[key] = plaintext

    def save_file(self, key: str, output_path: str):
        """将字典中的字节数据保存到文件"""
        if key not in self._data:
            raise KeyError(f"{key} not found in data")
        
        with open(output_path, "wb") as f:
            f.write(self._data[key])
        print(f"数据已保存到: {output_path}")
        
    def read_data(self, key: str):
        if key not in self._data:
            raise KeyError(f"{key} not found in data")
        
        return self._data[key]
    
    def __getitem__(self, key: str):
        return self.read_data(key)

    def save(self, output_path: str):
        """
        将 self._data 加密保存成 .edb 文件
        """
        try:
            password = 'Qz8mmWz2VEQ6X5Ic'
            # 生成密钥
            salt = os.urandom(16)  # 盐值
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=default_backend())
            key = kdf.derive(password.encode())  # 从密码派生密钥
            iv = os.urandom(12)  # 生成随机的初始化向量 (IV)
            encryptor = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend()).encryptor()
        except:
            pass
        
        # 序列化
        data_bytes = pickle.dumps(self._data, protocol=4)

        # 压缩数据
        if self._compress:
            data_bytes = zlib.compress(data_bytes)
        
        # 加密数据
        ciphertext = encryptor.update(data_bytes) + encryptor.finalize()
                
        with open(output_path, "wb") as f:
            f.write(salt + iv + encryptor.tag + ciphertext)  # 保存盐值、IV、TAG 和密文

        print(f"数据已加密并保存到: {output_path}")
        
    def load(self, input_path: str, password: str):
        """
        加载 .edb 文件为 self._data
        """
        with open(input_path, "rb") as f:
            try:
                password = password
                salt = f.read(16)
                iv = f.read(12)
                tag = f.read(16)
                ciphertext = f.read()
                kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=default_backend())
                key = kdf.derive(password.encode())
                decryptor = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend()).decryptor()
            except:
                pass
        # 解密数据
        data_bytes = decryptor.update(ciphertext) + decryptor.finalize()

        # 解压缩
        if self._compress:
            data_bytes = zlib.decompress(data_bytes)

        # 反序列化数据
        self._data = pickle.loads(data_bytes)      
    
    def __repr__(self):
        repr_str = f"EncryptDB(compress={self._compress}, {len(self._data)} items)"
        for key, item in self._data.items():
            if isinstance(item, bytes):
                repr_str += f"\n  - {key} : {len(item) / 1000:.0f} KB"
            else:
                repr_str += f"\n  - {key} : {item}"        
        
        return repr_str
    
    # --- 不同框架的模型加载方法 ---
    def load_to_onnx(self, key: str, use_gpu: bool=True):
        """
        从字典中加载指定的键，并将其导入为 ONNX session
        """
        if key not in self._data:
            raise KeyError(f"{key} not found")
        
        import onnxruntime
        from onnxruntime import set_default_logger_severity
        set_default_logger_severity(4)
        if use_gpu:
            session = onnxruntime.InferenceSession(self._data[key], providers=[ 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        else:
            session = onnxruntime.InferenceSession(self._data[key], providers=[ 'CPUExecutionProvider'])
   
        return session
    
    def load_to_torch(self, key: str):
        """
        torch 模型需要存两个文件, 一个是权重文件, key 为模型 id, 另一个是配置文件, key + ".config"
        """
        if key+".config" not in self._data:
            raise KeyError(f"{key} not found")

        from xensesdk.xenseInfer.nn_block import BaseModel
        
        model = BaseModel(self._data[key+".config"], str_byte=True)
        model.to("cuda")

        if key+".is_fused" in self._data:
            model.fuse()
            
        if "fp16" in key:
            model.half()

        model.load(io.BytesIO(self._data[key]))
        return model

    def load_to_rknn(self, key: str):
        """
        从字典中加载指定的键，并将其导入为 RKNN session
        """
        if key not in self._data:
            raise KeyError(f"{key} not found")
        
        from rknn.api import RKNN
        import tempfile
        
        model = RKNN()
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(self._data[key])
            model.load_rknn(temp_file.name)
            model.init_runtime(target='rk3576')

        return model
        
