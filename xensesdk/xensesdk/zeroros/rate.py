import time
from .utils.general_utils import accurate_sleep

def sleep(duration):
    if duration < 0.0:
        return
    else:
        accurate_sleep(duration)

class Rate:
    def __init__(self, frequency):
        self.period = 1.0 / frequency
        self.last_time = time.monotonic()  # 使用单调时钟避免时间跳变

    def sleep(self):
        current_time = time.monotonic()
        elapsed = current_time - self.last_time
        sleep_time = max(0, self.period - elapsed)
        accurate_sleep(sleep_time)
        self.last_time = current_time + sleep_time  # 补偿误差