import warnings
import functools


def singleton(cls):
    """
    单例装饰器，更优雅的实现

    Usage:
        @singleton
        class MyClass:
            pass
    """
    instances = {}

    def new(cls, *args, **kwargs):
        if cls not in instances:
            instances[cls] = object.__new__(cls)

        return instances[cls]

    cls.__new__ = new
    return cls

def infer_singleton(cls):
    """
    special单例装饰器

    Usage:
        @singleton
        class MyClass:
            pass
    """
    cls.__instances = {}
    cls.__new = cls.__new__
    cls.__init = cls.__init__

    def new(cls, *args, **kwargs):
        arg_set = frozenset(args).union(kwargs.values())
        if arg_set not in cls.__instances:
            instance = cls.__new(cls)
            cls.__init(instance, *args, **kwargs)
            cls.__instances[arg_set] = instance
        return cls.__instances[arg_set]

    cls.__init__ = lambda self, *args, **kwargs: None
    cls.__new__ = new
    return cls

def deprecated(reason: str = None):
    """
    标记函数为已弃用的装饰器
    
    Args:
        reason: 弃用原因说明
    """
    def decorator(func):
        @functools.wraps(func)  # 保留原函数的元数据
        def wrapper(*args, **kwargs):
            message = f"函数 {func.__name__} 已弃用"
            if reason:
                message += f"。{reason}"
            warnings.warn(message, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator