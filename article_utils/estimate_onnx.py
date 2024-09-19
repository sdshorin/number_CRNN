import torch
import onnxruntime
import numpy as np
from numbers_generator import HandwrittenNumbersDataset
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from functools import partial
import random
import json
from config import get_custom_dataset_folder
# Constants
TEST_DATASET_SIZE = 1000
BATCH_SIZE = 32



from tokenizer import Tokenizer

def collate_fn(batch, tokenizer):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data = torch.stack(data, dim=0)
    labels_encoded = tokenizer.encode(labels)
    return data, labels_encoded


def load_data(tokenizer, seed, batch_size=64):
        
    mnist_test = MNIST(root='./data', train=False, download=True)
    test_dataset = HandwrittenNumbersDataset(
        custom_dataset_folder=get_custom_dataset_folder(),
        mnist_dataset=mnist_test,
        max_digits=5,
        length=TEST_DATASET_SIZE,
        include_leading_zeros=False,
        seed=seed,
        pre_generate=True,
        num_threads=1
    )

    from functools import partial
    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)

    batch_size = BATCH_SIZE
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn_with_tokenizer)
    return test_loader

def evaluate_onnx_model(tokenizer, model_path, test_loader):
    session = onnxruntime.InferenceSession(model_path)
    correct = 0
    total = 0

    for data, labels in test_loader:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        data = data.numpy()
        
        predictions = session.run([output_name], {input_name: data})[0]
        
        predictions = np.argmax(predictions, axis=2)
        predictions = predictions.transpose(1, 0)
        
        decoded_preds = tokenizer.decode(predictions.tolist(), merge_repeated=True)
        
        for pred, label in zip(decoded_preds, labels):
            decoded_label = [val for pair in zip(label, [0]*len(label)) for val in pair]
            decoded_label = tokenizer.decode([decoded_label], merge_repeated=True)[0]
            if pred == decoded_label:
                correct += 1
            # else:
            #     print(f"Prediction: {pred}, Label: {label}, Decoded Label: {decoded_label}")
            total += 1

    accuracy = correct / total
    return accuracy

# Основная функция
def main():
    seed = 48
    random.seed(seed)
    torch.manual_seed(seed)

    alphabet = "0123456789"
    tokenizer = Tokenizer(alphabet)

    test_loader = load_data(tokenizer, seed)
    
    model_paths = [
        "models/crnn_OptimizedCRNN.onnx",
        "models/crnn_OptimizedCRNN_quantized.onnx",
        "models/crnn_SmallCRNN.onnx",
        "models/crnn_SmallCRNN_quantized.onnx",
        "models/crnn_OriginalCRNN.onnx",
        "models/crnn_OriginalCRNN_quantized.onnx",
    ]

    for i, model_path in enumerate(model_paths, 1):
        accuracy = evaluate_onnx_model(tokenizer, model_path, test_loader)
        print(f"Model {i} Accuracy: {accuracy:.4f}, Model Path: {model_path}")

if __name__ == "__main__":
    main()
