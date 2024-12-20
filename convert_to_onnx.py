import os
import torch
import onnx 
import onnxslim

current_directory = os.path.dirname(os.path.abspath(__file__))

pt_model_path = os.path.join(current_directory,"runs","detect","train","weights","best.pt")
onnx_model_path = os.path.join(current_directory,"onnx_models","best.onnx")

model = torch.load(pt_model_path)['model'].float()
model.eval()

dummy_input = torch.randn(1,3,640,640)

torch.onnx.export(model, dummy_input, onnx_model_path)

model_onnx = onnx.load(onnx_model_path)
model_onnx = onnxslim.slim(model_onnx)
onnx.save(model_onnx, onnx_model_path)


