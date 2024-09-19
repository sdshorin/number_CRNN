# main.py

DATASET_SIZE = 120_000
TEST_DATASET_SIZE = 3_000
BATCH_SIZE = 64
NUM_EPOCH = 75

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

from numbers_generator import HandwrittenNumbersDataset
from models import OriginalCRNN, OptimizedCRNN, SmallCRNN
from config import get_custom_dataset_folder
from tokenizer import Tokenizer, CTC_BLANK

# Вспомогательные функции
def train(model, criterion, optimizer, train_loader, device, tokenizer):
    model.train()
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)

        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets = torch.cat([torch.tensor(t, dtype=torch.long) for t in targets]).to(device)
        input_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=torch.long)

        loss = criterion(output, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # if batch_idx % 100 == 0:
        #     print(f'Train Batch {batch_idx}, Loss: {loss.item()}')

    return total_loss / len(train_loader)


def evaluate(model, tokenizer, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels_encoded in test_loader:

            data = data.to(device)
            output = model(data)
            output = output.cpu()
            preds = output.argmax(dim=2).transpose(1, 0)
            preds = preds.tolist()
            decoded_preds = tokenizer.decode(preds, merge_repeated=True)
            decoded_labels = tokenizer.decode(labels_encoded, merge_repeated=False)
            for pred, label in zip(decoded_preds, decoded_labels):
                if pred == label:
                    correct += 1
                total += 1
    accuracy = correct / total
    return accuracy


def collate_fn(batch, tokenizer):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data = torch.stack(data, dim=0)
    labels_encoded = tokenizer.encode(labels)
    return data, labels_encoded

def train_model(model, tokenizer):
    is_in_cloud = False
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        is_in_cloud = True
    else:
        device = torch.device("cpu")

    
    model.to(device)

    mnist_train = MNIST(root='./data', train=True, download=True)
    mnist_test = MNIST(root='./data', train=False, download=True)


    custom_dataset_folder = get_custom_dataset_folder()
    if not os.path.exists(custom_dataset_folder):
        custom_dataset_folder = "/home/user/sorted"
    
    train_dataset = HandwrittenNumbersDataset(
        custom_dataset_folder=custom_dataset_folder,
        mnist_dataset=mnist_train,
        max_digits=5,
        length=DATASET_SIZE,
        include_leading_zeros=False,
        seed=42,
        pre_generate=True,
        num_threads=1
    )

    test_dataset = HandwrittenNumbersDataset(
        custom_dataset_folder='',
        mnist_dataset=mnist_test,
        max_digits=5,
        length=TEST_DATASET_SIZE,
        include_leading_zeros=False,
        seed=43,
        pre_generate=True,
        num_threads=1
    )

    from functools import partial
    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)

    batch_size = BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn_with_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_with_tokenizer)


    criterion = nn.CTCLoss(blank=tokenizer.char_map[CTC_BLANK], zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Start training")
    print(f"Train params: dataset_size={DATASET_SIZE}, test_dataset_size={TEST_DATASET_SIZE}, batch_size={BATCH_SIZE}, num_epochs={NUM_EPOCH}")

    import time
    num_epochs = NUM_EPOCH
    for epoch in range(1, num_epochs + 1):
        train_start = time.time()
        train_loss = train(model, criterion, optimizer, train_loader, device, tokenizer)
        train_end = time.time()
        val_accuracy = evaluate(model, tokenizer, test_loader, device)
        train_accuracy = evaluate(model, tokenizer, train_loader, device)
        eval_end = time.time()
        estimated_end_time = train_start + (eval_end - train_start) * (num_epochs - epoch)
        human_readable_eta = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(estimated_end_time))
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Time: {train_end - train_start:.2f}, Eval Time: {eval_end - train_end:.2f}, ETA: {human_readable_eta}')

    return model


is_in_cloud = torch.cuda.is_available()
if __name__ == "__main__":

    # model = OriginalCRNN
    model = OptimizedCRNN
    # model = SmallCRNN

    if torch.backends.mps.is_available():
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            raise Exception("Please run 'export PYTORCH_ENABLE_MPS_FALLBACK=1' before running this script")

    alphabet = "0123456789"
    tokenizer = Tokenizer(alphabet)

    model = OptimizedCRNN(imgH=32, nc=1, nclass=len(tokenizer.char_map), nh=256, n_rnn=2, leakyRelu=False)
    print(f"Train {model.__class__.__name__}")
    trained_model = train_model(model, tokenizer)
    os.makedirs('models', exist_ok=True)
    torch.save(trained_model.state_dict(), f'models/crnn_{model.__class__.__name__}.pth')
