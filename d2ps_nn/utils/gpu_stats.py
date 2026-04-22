import torch
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetUUID,
    nvmlDeviceGetSerial,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetClockInfo,
    nvmlDeviceGetMemoryInfo,
    NVML_TEMPERATURE_GPU,
    NVML_CLOCK_SM,
    NVML_CLOCK_MEM,
)


class TorchGPUStats:
    def __init__(self, device="cuda:0"):
        nvmlInit()
        self._nvml_active = True

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"Expected CUDA device, got {self.device!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")

        self.torch_index = (
            torch.cuda.current_device()
            if self.device.index is None
            else self.device.index
        )

        self.torch_uuid = self._normalize_uuid(
            torch.cuda.get_device_properties(self.torch_index).uuid
        )
        self.handle, self.nvml_index, self.nvml_uuid = self._get_nvml_match()

    @staticmethod
    def _normalize_uuid(uuid_val):
        if isinstance(uuid_val, bytes):
            uuid_val = uuid_val.decode("utf-8")
        return str(uuid_val).strip().lower()

    def _get_nvml_match(self):
        for i in range(nvmlDeviceGetCount()):
            handle = nvmlDeviceGetHandleByIndex(i)
            nvml_uuid = self._normalize_uuid(nvmlDeviceGetUUID(handle))

            # direct equality first
            if nvml_uuid == self.torch_uuid:
                return handle, i, nvml_uuid

            # fallback in case torch gives shortened UUID form
            if nvml_uuid.endswith(self.torch_uuid) or self.torch_uuid.endswith(nvml_uuid):
                return handle, i, nvml_uuid

        raise RuntimeError(
            f"No NVML device matched torch device {self.device} "
            f"(torch UUID: {self.torch_uuid})"
        )

    def describe(self):
        name = nvmlDeviceGetName(self.handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        try:
            serial = nvmlDeviceGetSerial(self.handle)
            if isinstance(serial, bytes):
                serial = serial.decode("utf-8")
        except Exception:
            serial = None

        return {
            "torch_device": str(self.device),
            "torch_index": self.torch_index,
            "torch_uuid": self.torch_uuid,
            "nvml_index": self.nvml_index,
            "nvml_uuid": self.nvml_uuid,
            "name": name,
            "serial": serial,
        }

    def get_stats(self):
        util = nvmlDeviceGetUtilizationRates(self.handle)
        temp = nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)
        power_w = nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        sm_clock = nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_SM)
        mem_clock = nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_MEM)
        mem = nvmlDeviceGetMemoryInfo(self.handle)

        return {
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "temp_C": temp,
            "power_W": power_w,
            "sm_clock_MHz": sm_clock,
            "mem_clock_MHz": mem_clock,
            "mem_used_MB": mem.used / 1024**2,
            "mem_total_MB": mem.total / 1024**2,
            "mem_free_MB": mem.free / 1024**2,
        }

    def __del__(self):
        try:
            if getattr(self, "_nvml_active", False):
                nvmlShutdown()
        except Exception:
            pass