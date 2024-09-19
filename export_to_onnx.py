import torch
import onnx
import os
from onnxruntime.quantization import quantize_dynamic, QuantType
from models import OriginalCRNN, OptimizedCRNN, SmallCRNN

def export_to_onnx(model, file_path, input_shape):
    model.eval()
    dummy_input = torch.randn(input_shape, requires_grad=False)
    
    torch.onnx.export(model, 
                      dummy_input, 
                      file_path, 
                      export_params=True, 
                      opset_version=11, 
                      do_constant_folding=True, 
                      input_names=['input'], 
                      output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size', 3: 'width'}, 
                                    'output': {0: 'batch_size', 1: 'sequence'}})
    
    print(f"Model exported to {file_path}")

def quantize_onnx_model(input_path, output_path):
    quantize_dynamic(model_input=input_path,
                     model_output=output_path,
                     weight_type=QuantType.QUInt8)
    
    print(f"Quantized model saved to: {output_path}")
    
    quantized_model = onnx.load(output_path)
    onnx.checker.check_model(quantized_model)
    print(f"Quantized model verified successfully for {output_path}")

def process_model(model_class, model_name, input_shape=(1, 1, 32, 128)):

    model = model_class(imgH=32, nc=1, nclass=12, nh=256, n_rnn=2, leakyRelu=False)
    model_path = f'./models/{model_name}.pth'
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    onnx_model_path = f'./models/{model_name}.onnx'
    
    export_to_onnx(model, onnx_model_path, input_shape)
    
    quantized_onnx_model_path = f'./models/{model_name}_quantized.onnx'
    
    quantize_onnx_model(onnx_model_path, quantized_onnx_model_path)



models_info = [
    (OriginalCRNN, 'crnn_OriginalCRNN'),
    (OptimizedCRNN, 'crnn_OptimizedCRNN'),
    (SmallCRNN, 'crnn_SmallCRNN')
]
if __name__ == '__main__':
    
    for model_class, model_name in models_info:
        try:
            process_model(model_class, model_name)
        except Exception as e:
            print(f"An error occurred with model {model_name}: {str(e)}")

