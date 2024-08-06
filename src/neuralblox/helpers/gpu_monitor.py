import time
import GPUtil

class GPUMonitor:
    def __init__(self, gpu_index=0):
        self.gpu_index = gpu_index
        self.memory_usages = []

    def update_memory_usage(self):
        gpus = GPUtil.getGPUs()
        if gpus and len(gpus) > self.gpu_index:
            gpu = gpus[self.gpu_index]
            self.memory_usages.append(gpu.memoryUsed)

    def get_max_memory_usage(self):
        return max(self.memory_usages) if self.memory_usages else 0

    def get_avg_memory_usage(self):
        return sum(self.memory_usages) / len(self.memory_usages) if self.memory_usages else 0

def monitor_gpu_usage(gpu_monitor, duration=10, interval=1):
    start_time = time.time()
    while time.time() - start_time < duration:
        gpu_monitor.update_memory_usage()
        time.sleep(interval)
    max_memory_usage = gpu_monitor.get_max_memory_usage()
    avg_memory_usage = gpu_monitor.get_avg_memory_usage()
    return max_memory_usage, avg_memory_usage

if __name__ == "__main__":
    gpu_monitor = GPUMonitor(gpu_index=0)  # Monitor GPU 0
    max_memory, avg_memory = monitor_gpu_usage(gpu_monitor, duration=10, interval=1)  # Monitor for 60 seconds
    print(f"Max Memory Usage: {max_memory:.2f} MB")
    print(f"Avg Memory Usage: {avg_memory:.2f} MB")
