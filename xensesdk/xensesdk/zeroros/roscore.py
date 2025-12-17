import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .service import Service
from .timer import Timer
from .utils import ip_tool

@dataclass
class NodeInfo:
    ip_pid: str  # IP-PID combination, for identifying the node
    name: str
    services: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    last_heartbeat: Optional[float] = None    
    
@dataclass
class ServiceInfo:
    name: str
    ip: str
    port: int
    action_keys: List[str] = field(default_factory=list)

@dataclass
class TopicInfo:
    name: str
    ip: str
    port: int


class RosMaster(Service):
    
    NodeRegistry: Dict[str, NodeInfo] = {}
    ServiceRegistry: Dict[str, ServiceInfo] = {}
    TopicRegistry: Dict[str, TopicInfo] = {}
    ParameterRegistry: Dict[str, Any] = {}
    
    def __init__(self, port: int = 11411):
        super().__init__(port=port, timeout_ms=1000)
        
        self.timer = Timer(interval_ms=2500, callback=self.check_heartbeat)
        print(f"Zeroros master start at: {ip_tool.get_master_ip()}:{self.port}")
        # self.start(threaded=False)
            
    def impl_callback(self, payload):
        action_key = payload.get("action_key")
        
        if action_key in [
            "init_node", "del_node", "heartbeat",
            "add_service", "get_service", "del_service",
            "add_topic", "get_topic", "del_topic",
            "get_parameter", "set_parameter", "del_parameter",
            'list_topic', 'list_service', 'list_node', 'list_node_name', 'list_parameter'
        ]:
            return getattr(self, action_key)(*payload["args"], **payload["kwargs"])
        else:
            return {"success": False, "message": f"{action_key} not supported."}
    
    def check_heartbeat(self):
        """
        Checks the heartbeat of all nodes in the registry. If a node has not sent a heartbeat in the last 10 seconds,
        it is considered inactive and removed from the registry.
        """
        current_time = time.time()
        for ip_pid, node_info in list(self.NodeRegistry.items()):  # NOTE: must use list() to avoid RuntimeError caused by dictionary size change during iteration
            if node_info.last_heartbeat is None or current_time - node_info.last_heartbeat <= 5:
                continue

            print(f"Node {node_info.name} is inactive, removing from registry.")
            self.del_node(ip_pid)
    
    def heartbeat(self, ip_pid: str):
        """
        Updates the last heartbeat timestamp for the given node.
        """
        if ip_pid in self.NodeRegistry:
            self.NodeRegistry[ip_pid].last_heartbeat = time.time()
            return {"success": True}
        else:
            return {"success": False, "message": "Node not found."}        
    
    def init_node(self, ip_pid: str, name: str):
        """
        Registers a node with the given IP-PID combination.
        """
        if ip_pid in self.NodeRegistry:
            return {"success": False, "message": "Node already registered."}
        
        self.NodeRegistry[ip_pid] = NodeInfo(ip_pid, name=name)
        print(f"Node {name} registered.")
        return {"success": True}
    
    def del_node(self, ip_pid: str) -> dict:
        """
        Deletes a node with the given IP-PID combination.
        """
        if ip_pid in self.NodeRegistry:
            node_info = self.NodeRegistry[ip_pid]
            for service in node_info.services:
                self.del_service(service)
            for topic in node_info.topics:
                self.del_topic(topic)
            del self.NodeRegistry[ip_pid]
            return {"success": True}
        else:
            return {"success": False, "message": "Node not found."}
        
    def add_service(self, name: str, ip_pid: str, port: int) -> dict:
        """
        Registers a service with the given name, IP address, and port.
        """
        if name in self.ServiceRegistry:
            return {"success": False, "message": "Service already registered."}
        
        if ip_pid not in self.NodeRegistry:
            return {"success": False, "message": "Node not found."}
        
        ip = ip_pid.split("-")[0]
        self.ServiceRegistry[name] = ServiceInfo(name, ip, port)
        self.NodeRegistry[ip_pid].services.append(name)  
        return {"success": True}
        
    def del_service(self, name: str) -> dict:
        """
        Deletes a service with the given name.
        """
        if name in self.ServiceRegistry:
            self.ServiceRegistry.pop(name)
            return {"success": True}
        else:
            return {"success": False, "message": "Service not found."}
    
    def get_service(self, name: str) -> dict:
        """ 
        Gets the information of a service with the given name.
        """
        if name in self.ServiceRegistry:
            service_info = self.ServiceRegistry[name]
            return {
                "success": True,
                "name": service_info.name,
                "ip": service_info.ip,
                "port": service_info.port
            }
        else:
            return {"success": False, "message": "Service not found."}
        
    def add_topic(self, name: str, ip_pid: str, port: int) -> dict:
        """
        Registers a topic with the given name, IP address, and port.
        """
        if name in self.TopicRegistry:
            return {"success": False, "message": "Topic already registered."}

        if ip_pid not in self.NodeRegistry:
            return {"success": False, "message": "Node not found."}

        ip = ip_pid.split("-")[0]
        self.TopicRegistry[name] = TopicInfo(name, ip, port)
        self.NodeRegistry[ip_pid].topics.append(name)
        return {"success": True}
    
    def del_topic(self, name: str) -> dict:
        """
        Deletes a topic with the given name.
        """
        if name in self.TopicRegistry:
            self.TopicRegistry.pop(name)
            return {"success": True}
        else:
            return {"success": False, "message": "Topic not found."}
    
    def get_topic(self, name: str) -> dict:
        """ 
        Gets the information of a topic with the given name.
        """
        if name in self.TopicRegistry:
            topic_info = self.TopicRegistry[name]
            return {
                "success": True,
                "name": topic_info.name,
                "ip": topic_info.ip,
                "port": topic_info.port
            }
        else:
            return {"success": False, "message": "Topic not found."}
    
    def set_parameter(self, name: str, value: Any) -> dict:
        """
        Sets a parameter with the given name and value.
        """
        self.ParameterRegistry[name] = value
        return {"success": True}
    
    def del_parameter(self, name: str) -> dict:
        """
        Deletes a parameter with the given name.
        """
        if name in self.ParameterRegistry:
            self.ParameterRegistry.pop(name)
            return {"success": True}
        else:
            return {"success": False, "message": "Parameter not found."}
    
    def get_parameter(self, name: str) -> dict:
        """ 
        Gets the value of a parameter with the given name.
        """
        if name in self.ParameterRegistry:
            return {
                "success": True,
                "value": self.ParameterRegistry[name]
            }
        else:
            return {"success": False, "message": "Parameter not found."}
    
    def list_topic(self) -> dict:
        """
        Lists all registered topics.
        """
        return {
            "success": True,
            "topics": list(self.TopicRegistry.keys())
        }
    
    def list_service(self) -> dict:
        """
        Lists all registered services.
        """
        return {
            "success": True,
            "services": list(self.ServiceRegistry.keys())
        }
    
    def list_node(self) -> dict:
        """
        Lists all registered nodes.
        """
        return {
            "success": True,
            "nodes": list(self.NodeRegistry.keys())
        }

    def list_node_name(self) -> dict:
        """
        Lists all registered nodes.
        """
        return {
            "success": True,
            "nodes": list(node.name for node in self.NodeRegistry.values())
        }
    
    def list_parameter(self) -> dict:
        """
        Lists all registered parameters.
        """
        return {
            "success": True,
            "parameters": self.ParameterRegistry
        }