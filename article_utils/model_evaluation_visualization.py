import torch
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
from numbers_generator import HandwrittenNumbersDataset
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from functools import partial
import random

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

    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn_with_tokenizer)
    return test_loader

def evaluate_onnx_model(tokenizer, model_path, test_loader):
    session = onnxruntime.InferenceSession(model_path)
    correct_samples = []
    incorrect_samples = []

    for data, labels in test_loader:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        data_np = data.numpy()
        predictions = session.run([output_name], {input_name: data_np})[0]
        
        predictions = np.argmax(predictions, axis=2)
        predictions = predictions.transpose(1, 0)
        
        decoded_preds = tokenizer.decode(predictions.tolist(), merge_repeated=True)
        
        for i, (pred, label) in enumerate(zip(decoded_preds, labels)):
            decoded_label = [val for pair in zip(label, [0]*len(label)) for val in pair]
            decoded_label = tokenizer.decode([decoded_label], merge_repeated=True)[0]
            
            if pred == decoded_label:
                correct_samples.append((data[i], pred, decoded_label))
            else:
                incorrect_samples.append((data[i], pred, decoded_label))

    return correct_samples, incorrect_samples

def plot_samples(correct_samples, incorrect_samples, output_path):
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axs = plt.subplots(4, 3, figsize=(15, 10))
    fig.suptitle("Примеры работы модели OptimizedCRNN после квантизации", fontsize=16)

    for i in range(6):
        ax = axs[i // 3, i % 3]
        image, pred, label = correct_samples[i]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f"Прогноз: {pred}, Факт: {label}", fontsize=10)
        ax.axis('off')

    fig.text(0.5, 0.93, "Корректно распознанные образцы", ha='center', fontsize=14)

    for i in range(6):
        ax = axs[(i + 6) // 3, (i + 6) % 3]
        image, pred, label = incorrect_samples[i]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f"Прогноз: {pred}, Факт: {label}", fontsize=10)
        ax.axis('off')

    fig.text(0.5, 0.48, "Ошибочно распознанные образцы", ha='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    SEED = 87
    random.seed(SEED)
    torch.manual_seed(SEED)
    alphabet = "0123456789"
    tokenizer = Tokenizer(alphabet)
    model_path = "results/model_1_middle/crnn_model_quantized.onnx"
    test_loader = load_data(tokenizer, SEED)
    
    correct_samples, incorrect_samples = evaluate_onnx_model(tokenizer, model_path, test_loader)
    
    plot_samples(correct_samples, incorrect_samples, './images/model_evaluation_results.png')
    print(f"Результаты сохранены в './images/model_evaluation_results.png'")

if __name__ == "__main__":
    main()
