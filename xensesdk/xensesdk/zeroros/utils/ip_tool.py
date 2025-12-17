import os
import socket
import psutil

def get_free_port() -> int:
    """
    Get a free port on the local machine.
    :return: A free port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # 0 代表让操作系统自动分配端口
        return s.getsockname()[1]  # 返回分配的端口号

# def get_all_ipv4_addresses() -> list:
#     host_name = socket.gethostname()
#     all_ips = socket.getaddrinfo(host_name, None)
#     # 过滤出 IPv4 地址
#     ipv4_addresses = [ip[4][0] for ip in all_ips if ip[0] == socket.AF_INET]
#     return ipv4_addresses

def get_all_ipv4_addresses():
    ips = []
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                ips.append(addr.address)
    return ips

def get_local_ip(env_name="ZEROROS_IP") -> str:
    """
    获取本机的 IP 地址, 环境变量 ZEROROS_IP 优先级最高, 其次是获取本机的 IPv4 地址的第一项
    """
    ip = os.getenv(env_name)
    if ip:
        return ip
    all_ipv4_addresses = get_all_ipv4_addresses()
    if all_ipv4_addresses:
        return all_ipv4_addresses[0]
    raise Exception("No valid IP address found")

def get_master_ip() -> str:
    """
    获取主机的 IP 地址, 环境变量 ZEROROS_MASTER_IP 优先级最高, 其次是获取本机的 IPv4 地址的第一项
    """
    return get_local_ip("ZEROROS_MASTER_IP")
    
def is_same_subnet(ip1: str, ip2: str) -> bool:
    """
    简单判断两个 IP 地址是否属于同一个局域网 ( 假设子网掩码为 255.255.255.0 )
    :param ip1: 第一个 IP 地址
    :param ip2: 第二个 IP 地址
    :return: 如果属于同一个局域网, 返回 True, 否则返回 False
    """
    return ip1.split('.')[:3] == ip2.split('.')[:3]