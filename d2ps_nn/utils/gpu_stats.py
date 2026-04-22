import torch
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetUUID,
    nvmlDeviceGetSerial,
    nvmlDeviceGetPciInfo,
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
    """
    Resolve an NVML handle from a torch device once, then reuse it.

    Example
    -------
    stats = TorchGPUStats("cuda:0")
    print(stats.describe())
    print(stats.get_stats())
    """

    def __init__(self, device="cuda:0"):
        nvmlInit()

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"Expected a CUDA device, got {self.device!r}")

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")

        self.torch_index = self._resolve_torch_index(self.device)
        self.bus_id = self._get_torch_bus_id(self.torch_index)
        self.handle = self._get_nvml_handle_from_bus_id(self.bus_id)

    @staticmethod
    def _resolve_torch_index(device: torch.device) -> int:
        # torch.device("cuda") can have index=None; treat that as current device
        if device.index is None:
            return torch.cuda.current_device()
        return device.index

    @staticmethod
    def _normalize_bus_id(bus_id) -> str:
        # torch returns a string like '00000000:17:00.0'
        # NVML may return bytes or str depending on binding/version
        if isinstance(bus_id, bytes):
            bus_id = bus_id.decode("utf-8")
        return bus_id.strip().lower()

    def _get_torch_bus_id(self, torch_index: int) -> str:
        bus_id = torch.cuda.get_device_properties(torch_index).pci_bus_id
        return self._normalize_bus_id(bus_id)

    def _get_nvml_handle_from_bus_id(self, target_bus_id: str):
        n = nvmlDeviceGetCount()

        for nvml_index in range(n):
            handle = nvmlDeviceGetHandleByIndex(nvml_index)
            pci = nvmlDeviceGetPciInfo(handle)
            nvml_bus_id = self._normalize_bus_id(pci.busId)

            if nvml_bus_id == target_bus_id:
                self.nvml_index = nvml_index
                return handle

        raise RuntimeError(
            f"Could not find NVML handle matching torch bus ID {target_bus_id!r}"
        )

    def describe(self) -> dict:
        name = nvmlDeviceGetName(self.handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        uuid = nvmlDeviceGetUUID(self.handle)
        if isinstance(uuid, bytes):
            uuid = uuid.decode("utf-8")

        try:
            serial = nvmlDeviceGetSerial(self.handle)
            if isinstance(serial, bytes):
                serial = serial.decode("utf-8")
        except Exception:
            serial = None

        return {
            "torch_device": str(self.device),
            "torch_index": self.torch_index,
            "nvml_index": self.nvml_index,
            "name": name,
            "bus_id": self.bus_id,
            "uuid": uuid,
            "serial": serial,
        }

    def get_stats(self) -> dict:
        util = nvmlDeviceGetUtilizationRates(self.handle)
        temp = nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)
        power_w = nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        sm_clock = nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_SM)
        mem_clock = nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_MEM)
        mem = nvmlDeviceGetMemoryInfo(self.handle)

        return {
            "gpu_util": util.gpu,                    # percent
            "mem_util": util.memory,                # percent
            "temp_C": temp,
            "power_W": power_w,
            "sm_clock_MHz": sm_clock,
            "mem_clock_MHz": mem_clock,
            "mem_used_MB": mem.used / 1024**2,
            "mem_total_MB": mem.total / 1024**2,
            "mem_free_MB": mem.free / 1024**2,
        }

    def shutdown(self):
        nvmlShutdown()

    def __del__(self):
        # Best-effort cleanup
        try:
            nvmlShutdown()
        except Exception:
            pass
