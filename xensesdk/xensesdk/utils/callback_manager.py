
class CallbackManager:
    def __init__(self):
        self.callbacks  = []

    def register_callback(self,  callback):
        """注册一个回调函数
        Args:
            callback: 回调函数
        """
        self.callbacks.append(callback)
    
    def _notify_callbacks(self):
        """触发所有回调函数
        """
        for callback in self.callbacks:
            callback()
