import time
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

class GPUMonitor:
    def __init__(self, gpu_index=0):
        self.gpu_index = gpu_index
        self.memory_usages = []
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(gpu_index)

    def update_memory_usage(self):
        info = nvmlDeviceGetMemoryInfo(self.handle)
        self.memory_usages.append(info.used / (1024 ** 2))  # Convert to MB
        
    def reset(self):
        self.memory_usages = []

    def get_max_memory_usage(self):
        return max(self.memory_usages) if self.memory_usages else 0

    def get_avg_memory_usage(self):
        return sum(self.memory_usages) / len(self.memory_usages) if self.memory_usages else 0

    def shutdown(self):
        nvmlShutdown()

def monitor_gpu_usage(gpu_monitor, duration=10, interval=1):
    start_time = time.time()
    while time.time() - start_time < duration:
        gpu_monitor.update_memory_usage()
        time.sleep(interval)
    max_memory_usage = gpu_monitor.get_max_memory_usage()
    avg_memory_usage = gpu_monitor.get_avg_memory_usage()
    gpu_monitor.shutdown()
    return max_memory_usage, avg_memory_usage

if __name__ == "__main__":
    gpu_monitor = GPUMonitor(gpu_index=0)  # Monitor GPU 0
    max_memory, avg_memory = monitor_gpu_usage(gpu_monitor, duration=10, interval=1)  # Monitor for 60 seconds
    print(f"Max Memory Usage: {max_memory:.2f} MB")
    print(f"Avg Memory Usage: {avg_memory:.2f} MB")
