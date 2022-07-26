import torch
import model4
from io import BytesIO

f = BytesIO()
model = model4.Policy_Value()
print(model)
dummy_input = torch.randn(128, 5, 15, 15)
torch.onnx.export(
    model,
    dummy_input,
    #"1.onnx",  # where to save the model (can be a file or file-like object)
    f,
    export_params=True,  # store the trained parameter weights inside the model file
    # do_constant_folding=False,
    # training=2,
    #operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    # verbose=True,
    opset_version=9,
    #keep_initializers_as_inputs=False,
    # export_modules_as_functions=True,
    input_names=["input"],
    output_names=["prob", "v"]
)
b = f.getvalue()
print(len(b), type(b))
# torch.onnx.export(
#     model,
#     dummy_input,
#     "1.onnx",  # where to save the model (can be a file or file-like object)
#     export_params=True,  # store the trained parameter weights inside the model file
#     opset_version=16,  # the ONNX version to export the model to
#     do_constant_folding=True,  # whether to execute constant folding for optimization
#     input_names=["input"],  # the model's input names
#     output_names=["output"],  # the model's output names
#     dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable length axes
# )

# model = model.half().to('cuda')
# dummy_input = dummy_input.to(torch.float16).to('cuda')
# torch.onnx.export(
#     model,
#     dummy_input,
#     "2.onnx",  # where to save the model (can be a file or file-like object)
#     export_params=True,  # store the trained parameter weights inside the model file
#     opset_version=16,  # the ONNX version to export the model to
#     do_constant_folding=True,  # whether to execute constant folding for optimization
#     input_names=["input"],  # the model's input names
#     output_names=["output"],  # the model's output names
#     dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable length axes
# )
