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

import torch.nn.functional as F

# class CRNN(nn.Module):
#     def __init__(self, imgH=32, nc=1, nclass=12, nh=256, n_rnn=2, leakyRelu=False):
#         super(CRNN, self).__init__()
#         assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

#         ks = [3, 3, 3, 3, 3, 3, 2]
#         ps = [1, 1, 1, 1, 1, 1, 0]
#         ss = [1, 1, 1, 1, 1, 1, 1]
#         nm = [64, 128, 256, 256, 512, 512, 512]

#         cnn = nn.Sequential()

#         def convRelu(i, batchNormalization=False):
#             nIn = nc if i == 0 else nm[i - 1]
#             nOut = nm[i]
#             cnn.add_module('conv{0}'.format(i),
#                            nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
#             if batchNormalization:
#                 cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
#             if leakyRelu:
#                 cnn.add_module('relu{0}'.format(i),
#                                nn.LeakyReLU(0.2, inplace=True))
#             else:
#                 cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

#         convRelu(0)
#         cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
#         convRelu(1)
#         cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
#         convRelu(2, True)
#         convRelu(3)
#         cnn.add_module('pooling{0}'.format(2),
#                        nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
#         convRelu(4, True)
#         convRelu(5)
#         cnn.add_module('pooling{0}'.format(3),
#                        nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
#         convRelu(6, True)  # 512x1x16

#         self.cnn = cnn
#         self.rnn = nn.Sequential(
#             BidirectionalLSTM(nm[-1], nh, nh),
#             BidirectionalLSTM(nh, nh, nclass))

#     def forward(self, input):
#         # conv features
#         conv = self.cnn(input)
#         b, c, h, w = conv.size()
#         assert h == 1, "the height of conv must be 1"
#         conv = conv.squeeze(2)
#         conv = conv.permute(2, 0, 1)  # [w, b, c]

#         # rnn features
#         output = self.rnn(conv)

#         # add log_softmax to converge output
#         output = F.log_softmax(output, dim=2)

#         return output

# class BidirectionalLSTM(nn.Module):
#     def __init__(self, nIn, nHidden, nOut):
#         super(BidirectionalLSTM, self).__init__()
#         self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
#         self.embedding = nn.Linear(nHidden * 2, nOut)

#     def forward(self, input):
#         recurrent, _ = self.rnn(input)
#         T, b, h = recurrent.size()
#         t_rec = recurrent.view(T * b, h)
#         output = self.embedding(t_rec)  # [T * b, nOut]
#         output = output.view(T, b, -1)
#         return output


class CRNN(nn.Module):
    def __init__(self, imgH=32, nc=1, nclass=12, nh=128, n_rnn=1, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [32, 64, 128, 128, 256, 256, 256]  # Уменьшенные количества каналов

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(6, True)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(nm[-1], nh, nclass))  # Уменьшено количество рекуррентных слоёв

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = self.rnn(conv)
        output = F.log_softmax(output, dim=2)
        return output


# Константы
CTC_BLANK = '<BLANK>'
OOV_TOKEN = '<OOV>'

class Tokenizer:
    def __init__(self, alphabet):
        self.char_map = {CTC_BLANK: 0, OOV_TOKEN: 1}
        for i, char in enumerate(alphabet):
            self.char_map[char] = i + 2
        self.rev_char_map = {v: k for k, v in self.char_map.items()}

    def encode(self, word_list):
        enc_words = []
        for word in word_list:
            enc_words.append([self.char_map.get(char, self.char_map[OOV_TOKEN]) for char in word])
        return enc_words

    def decode(self, enc_word_list, merge_repeated=True):
        dec_words = []
        for word in enc_word_list:
            word_chars = ''
            prev_char = None
            for char_enc in word:
                char = self.rev_char_map.get(char_enc, OOV_TOKEN)
                if char != CTC_BLANK and char != OOV_TOKEN:
                    if not (merge_repeated and char == prev_char):
                        word_chars += char
                prev_char = char
            dec_words.append(word_chars)
        return dec_words


# Вспомогательные функции
def train(model, criterion, optimizer, train_loader, device, tokenizer):
    model.train()
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print("Output shape:", output.shape)  # Проверка размерности выхода модели

        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets = torch.cat([torch.tensor(t, dtype=torch.long) for t in targets]).to(device)
        input_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=torch.long)

        # print("Target shape:", targets.shape)  # Проверка размерности целевых данных
        # print("Input lengths:", input_lengths)
        # print("Target lengths:", target_lengths)

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
    # print("Sample encoded label:", labels_encoded[0])  # Проверка закодированной метки
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

    # Определение алфавита и создание токенизатора
    
    model.to(device)

    # Загрузка датасета MNIST
    mnist_train = MNIST(root='./data', train=True, download=True)
    mnist_test = MNIST(root='./data', train=False, download=True)

    # Создание датасетов
    if not is_in_cloud:
        from numbers_generator import HandwrittenNumbersDataset

    custom_dataset_folder = '/Users/sergejsorin/work/math/lib/mnist_improve/local_data/sorted'
    if not os.path.exists(custom_dataset_folder):
        custom_dataset_folder = "/home/user/sorted"

    train_dataset = HandwrittenNumbersDataset(
        custom_dataset_folder=custom_dataset_folder,  # Путь к вашему кастомному датасету, если есть
        mnist_dataset=mnist_train,
        max_digits=5,
        length=DATASET_SIZE,
        include_leading_zeros=False,
        seed=42,
        pre_generate=False,
        num_threads=1
    )

    test_dataset = HandwrittenNumbersDataset(
        custom_dataset_folder='',
        mnist_dataset=mnist_test,
        max_digits=5,
        length=TEST_DATASET_SIZE,
        include_leading_zeros=False,
        seed=43,
        pre_generate=False,
        num_threads=1
    )

    from functools import partial
    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)

    # Создание DataLoader
    batch_size = BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn_with_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_with_tokenizer)


    # Определение критерия и оптимизатора
    criterion = nn.CTCLoss(blank=tokenizer.char_map[CTC_BLANK], zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Start training")
    print(f"Train params: dataset_size={DATASET_SIZE}, test_dataset_size={TEST_DATASET_SIZE}, batch_size={BATCH_SIZE}, num_epochs={NUM_EPOCH}")

    import time
    # Цикл обучения
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

    # Пример использования модели для предсказания
    model.eval()
    with torch.no_grad():
        # Предположим, у нас есть одно тестовое изображение из тестового датасета
        data_iter = iter(test_loader)
        test_image, test_label = next(data_iter)
        test_image = test_image[0].unsqueeze(0).to(device)
        output = model(test_image)
        pred = output.argmax(dim=2).transpose(1, 0)
        decoded_pred = tokenizer.decode(pred.tolist(), merge_repeated=True)[0]
        print(f"Predicted number: {decoded_pred}")
        print(f"Actual number: {test_label[0]}")
    return model


is_in_cloud = torch.cuda.is_available()
if __name__ == "__main__":
    if not is_in_cloud:
        
        alphabet = "0123456789"
        tokenizer = Tokenizer(alphabet)

        model = CRNN(imgH=32, nc=1, nclass=len(tokenizer.char_map), nh=256, n_rnn=2, leakyRelu=False)

        trained_model = train_model(model, tokenizer)
        os.makedirs('models', exist_ok=True)
        torch.save(trained_model.state_dict(), 'models/crnn_model.pth')
