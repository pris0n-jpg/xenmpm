import sys
import argparse
from xensesdk.zeroros import RosMaster, Node

def main():
    # 创建主解析器
    parser = argparse.ArgumentParser(description="ZeroROS command line tool")
    subparsers = parser.add_subparsers(dest="command", help="subcommands")

    # 启动 roscore
    parser_core = subparsers.add_parser("core", help="launch roscore")
    parser_core.add_argument("-p", "--port", type=int, default=11411, help="port for roscore")

    # topic
    parser_topic = subparsers.add_parser("topic", help="topic commands")
    parser_topic.add_argument("-l", "--list", action="store_true", help="list topics")
    
    # service
    parser_service = subparsers.add_parser("service", help="service commands")
    parser_service.add_argument("-l", "--list", action="store_true", help="list services")
    
    # node
    parser_node = subparsers.add_parser("node", help="node commands")
    parser_node.add_argument("-l", "--list", action="store_true", help="list nodes")
    
    # parameters
    parser_param = subparsers.add_parser("param", help="parameter commands")
    parser_param.add_argument("-l", "--list", action="store_true", help="list parameters")
    parser_param.add_argument("-s", "--set", type=str, help="set parameter, format: key:=value")
    parser_param.add_argument("-g", "--get", type=str, help="get parameter, format: key")
    
    # 解析命令行参数
    args = parser.parse_args()

    # 根据子命令调用相应的函数
    if args.command == "core":
        core = RosMaster(port=args.port)
        core.start(threaded = False)
    elif args.command == "topic":
        handle_topic(args)
    elif args.command == "service":
        handle_service(args)
    elif args.command == "node":
        handle_node(args)
    elif args.command == "param":
        handle_param(args)
    else:
        parser.print_help()

def handle_topic(args):
    if args.list:
        tmp_node = Node()
        print("Topics:")
        for topic in tmp_node.list_topic():
            print(f"- {topic}")

def handle_service(args):
    if args.list:
        tmp_node = Node()
        print("Services:")
        for service in tmp_node.list_service():
            print(f"- {service}")
    
def handle_node(args):
    if args.list:
        tmp_node = Node()
        tmp_node._MasterProxy.del_node(tmp_node._ip_pid)  # 删除临时节点的注册信息
        print("Nodes:")
        for node in tmp_node.list_node():
            print(f"- {node}")
        
def validate_value(value):
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit():
        return float(value)
    elif value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        return str(value)
    
def handle_param(args):
    tmp_node = Node()
    if args.list:
        print("Parameters:")
        for param, value in tmp_node.list_parameter().items():
            print(f"- {param} = {value}")
    elif args.set:
        key, value = args.set.split(":=")
        value = validate_value(value)
        tmp_node.set_parameter(key, value)
        print(f"Set parameter: {key} = {value}")
    elif args.get:
        key = args.get
        value = tmp_node.get_parameter(key)
        print(f"Get parameter: {key} = {value}")


if __name__ == "__main__":
    main()