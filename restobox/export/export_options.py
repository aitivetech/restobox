from dataclasses import dataclass


@dataclass(frozen=True)
class ExportOptions:
    export: bool = True
    optimize: bool = False
    quantize_fp16: bool = True
    quantize_int8: bool = False