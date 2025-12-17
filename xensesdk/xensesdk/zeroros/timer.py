import threading
import time

class Timer:
    def __init__(self, interval_ms, callback, start=True, delay_ms=0):
        self.interval = interval_ms / 1000.0  # 转换为秒
        self._delay_s = delay_ms / 1000.0  # 转换为秒
        self.callback = callback
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        if start:
            self._thread.start()

    def _run(self):
        self._stop_event.wait(self._delay_s)
        while not self._stop_event.is_set():
            start_time = time.monotonic()  # 使用单调时钟避免系统时间跳变
            try:
                self.callback()
            except Exception as e:
                print(f"Callback error: {e}")
            elapsed = time.monotonic() - start_time
            sleep_time = max(0, self.interval - elapsed)  # 强制非负等待
            self._stop_event.wait(sleep_time)
    def start(self):
        self._thread.start()

    def stop(self):
        if not self._stop_event.is_set():
            self._stop_event.set()
    
        if self._thread.is_alive():
            self._thread.join()
    
    def alive(self):
        return self._thread.is_alive()


if __name__ == "__main__":
            
    def example_callback():
        curr_time = time.time()
        print("Callback called at: ", curr_time)
    r = Timer(1.0, example_callback)
    try:
        while True:
            pass
    except KeyboardInterrupt:
        r.stop()
        print("Timer stopped")
