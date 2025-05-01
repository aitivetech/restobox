import onnx
import torch
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType

from restobox.export.export_utilities import create_dynamic_shape
from restobox.models.model import Model

def convert_int8_dynamic(onnx_fp32_path, onnx_int8_path):
    quantize_dynamic(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_path,
        weight_type=QuantType.QInt8  # You can also use QuantType.QUInt8
    )

def convert_fp16(onnx_fp32_path, onnx_fp16_path):
    model = onnx.load(onnx_fp32_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, onnx_fp16_path)

def export_onnx(model: Model,
                path: str,
                device: torch.device,
                dtype: torch.dtype,
                optimize: bool):
    input_names = [i[0] for i in model.inputs]
    output_names = [i[0] for i in model.outputs]

    dynamic_shapes = list()
    for idx,entry in enumerate(model.inputs):
        dynamic_shapes.append(create_dynamic_shape(entry[1], optimize))

    input_samples = [i[1].create_random(device,dtype=dtype) for i in model.inputs]

    if optimize:
        torch.onnx.export(model.root,
                          args=tuple(input_samples),
                          f=path,
                          dynamic_shapes=dynamic_shapes,
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=18,
                          verbose=False,
                          dynamo=True,
                          optimize=True)
    else:
        torch.onnx.export(model.root,
                          args=tuple(input_samples),
                          f=path,
                          dynamic_axes=dynamic_shapes,
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=18,
                          verbose=False,
                          dynamo=False)