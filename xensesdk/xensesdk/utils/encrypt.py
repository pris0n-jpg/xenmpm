from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import os 
import json

def encrypt_config(data, output_path, password):
    # 生成密钥
    salt = os.urandom(16)  # 盐值
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(password.encode())  # 从密码派生密钥

    # 生成随机的初始化向量 (IV)
    iv = os.urandom(12)

    # 设置 AES-GCM 加密
    encryptor = Cipher(
        algorithms.AES(key),
        modes.GCM(iv),
        backend=default_backend()
    ).encryptor()

    # 加密模型文件
    json_string = json.dumps(data)

    # 将 JSON 字符串编码为字节数据
    byte_data = json_string.encode('utf-8')
    ciphertext = encryptor.update(byte_data) + encryptor.finalize()

    # 保存加密数据
    with open(output_path, "wb") as f:
        f.write(salt + iv + encryptor.tag + ciphertext)  # 保存盐值、IV、TAG 和密文

    print(f"模型已并保存到: {output_path}")

def decrypt_config(encrypted_path, password):
    with open(encrypted_path, "rb") as f:
        salt = f.read(16)  # 读取盐值
        iv = f.read(12)    # 读取初始化向量 (IV)
        tag = f.read(16)   # 读取 GCM TAG
        ciphertext = f.read()  # 读取密文

    # 从密码派生密钥
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(password.encode())

    # 设置 AES-GCM 解密
    decryptor = Cipher(
        algorithms.AES(key),
        modes.GCM(iv, tag),
        backend=default_backend()
    ).decryptor()

    # 解密模型
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    loaded_data = json.loads(plaintext.decode('utf-8'))
    return loaded_data  # 返回解密后的路径
