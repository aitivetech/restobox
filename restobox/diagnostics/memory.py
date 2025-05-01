import torch


def get_memory_info(device: torch.device) -> tuple[int,int]:
    if device.type == "cuda":
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        return total_mem, free_mem

    return 0,0
