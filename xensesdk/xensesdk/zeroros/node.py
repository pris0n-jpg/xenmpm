import os
from . import Publisher, Subscriber, Timer, Service, ServiceProxy
from .utils import ip_tool

class Node:
    
    Cnt:int = 0  # 区分同一个进程中创建的多个节点
    
    def __init__(
        self, 
        master_ip: str = None,
        name: str = None
    ):
        self._master_ip = master_ip if master_ip else ip_tool.get_master_ip()
        self._local_ip = ip_tool.get_local_ip()
        self._pid = os.getpid()
        self._cnt = Node.Cnt
        Node.Cnt += 1
        self._ip_pid = f"{self._local_ip}-{self._pid}-{self._cnt}"
        self._name = name if name else f"Node-{self._ip_pid}"
                            
        self._Publisher = {}
        self._Subsriber = {}
        self._Service = {}
        self._ServiceProxy = {}
        self._Timer = []
        
        self._MasterProxy = ServiceProxy(
            ip=self._master_ip, 
            port=11411, 
        )

        ret = self._MasterProxy.init_node(self._ip_pid, self._name)        
        if not ret["success"]:
            self._MasterProxy.stop()
            raise ConnectionError(f"Failed to connect to master at {self._master_ip}:11411.")
        
        self.create_timer(interval_ms=2000, callback=self.heartbeat)
    
    def heartbeat(self):
        """
        Send heartbeat to master node to keep the connection alive.
        """
        self._MasterProxy.heartbeat(self._ip_pid)
            
    def create_publisher(self, topic: str, msg_class=object, hwm=20) -> Publisher:
        port = ip_tool.get_free_port()
        ret = self._MasterProxy.add_topic(topic, self._ip_pid, port)
        if not ret["success"]:
            raise ValueError(f"Create topic {topic} failed: {ret['message']}")
        
        self._Publisher[topic] = Publisher(topic, msg_class, port=port, hwm=hwm)
        return self._Publisher[topic]

    def stop_publisher(self, topic: str):
        self._Publisher[topic].stop()
        self._MasterProxy.del_topic(topic)
    
    def create_subscriber(self, topic: str, callback: callable, msg_class=object, hwm=20) -> Subscriber:
        topic_info = self._MasterProxy.get_topic(topic)
        if not topic_info["success"]:
            raise ValueError(f"Topic {topic} not found.")
        
        ip = topic_info["ip"]
        port = topic_info["port"]
        self._Subsriber[topic] = Subscriber(topic, msg_class, callback, ip=ip, port=port, hwm=hwm)
        return self._Subsriber[topic]
    
    def stop_subscriber(self, topic: str):
        self._Subsriber[topic].stop()
    
    def create_service(self, name: str, timeout_ms: int=800) -> Service:
        port = ip_tool.get_free_port()
        ret = self._MasterProxy.add_service(name, self._ip_pid, port)
        if not ret["success"]:
            raise ValueError(f"Create service {name} failed: {ret['message']}")
        
        self._Service[name] = Service(port=port, timeout_ms=timeout_ms)
        print("service needs manually start")
        return self._Service[name]
    
    def stop_service(self, name: str):
        self._Service[name].stop()
        self._MasterProxy.del_service(name)
    
    def create_client(self, name: str) -> ServiceProxy:
        service_info = self._MasterProxy.get_service(name)
        if not service_info["success"]:
            raise ValueError(f"Service {name} not found.")
        
        ip = service_info["ip"]
        port = service_info["port"]
        self._ServiceProxy[name] = ServiceProxy(ip=ip, port=port)
        return self._ServiceProxy[name]
    
    def stop_client(self, name: str):
        self._ServiceProxy[name].stop()

    def create_timer(self, interval_ms: int, callback: callable) -> Timer:
        timer = Timer(interval_ms, callback)
        self._Timer.append(timer)
        return timer

    def destroy_node(self):
        
        for pub in self._Publisher.values():
            pub.stop()
        for sub in self._Subsriber.values():
            sub.stop()
        for srv in self._Service.values():
            srv.stop()
        for cli in self._ServiceProxy.values():
            cli.stop()
        
        self._MasterProxy.del_node(self._ip_pid)
        self._MasterProxy.stop()
        
        for timer in self._Timer:
            timer.stop()
            
        print("node destroyed")
    
    def list_topic(self) -> list:
        ret = self._MasterProxy.list_topic()
        if not ret["success"]:
            return []
        else:
            return ret["topics"]
        
    def list_service(self) -> list:
        ret = self._MasterProxy.list_service()
        if not ret["success"]:
            return []
        else:
            return ret["services"]
    
    def list_node(self) -> list:
        ret = self._MasterProxy.list_node()
        if not ret["success"]:
            return []
        else:
            return ret["nodes"]
    
    def set_parameter(self, key: str, value: object):
        ret = self._MasterProxy.set_parameter(key, value)
        if not ret["success"]:
            print(f"Set parameter {key} failed: {ret['message']}")
        return ret["success"]
    
    def get_parameter(self, key: str):
        ret = self._MasterProxy.get_parameter(key)
        if not ret["success"]:
            print(f"Get parameter {key} failed: {ret['message']}")
            return None
        else:
            return ret["value"]
    
    def list_parameter(self) -> list:
        ret = self._MasterProxy.list_parameter()
        if not ret["success"]:
            return []
        else:
            return ret["parameters"]
    
