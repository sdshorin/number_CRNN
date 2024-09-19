
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output



class OriginalCRNN(nn.Module):
    def __init__(self, imgH=32, nc=1, nclass=12, nh=256, n_rnn=2, leakyRelu=False):
        super(OriginalCRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

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
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(nm[-1], nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        output = self.rnn(conv)

        output = F.log_softmax(output, dim=2)

        return output



class OptimizedCRNN(nn.Module):
    def __init__(self, imgH=32, nc=1, nclass=12, nh=128, n_rnn=1, leakyRelu=False):
        super(OptimizedCRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [32, 64, 128, 128, 256, 256, 256]

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


class SmallCRNN(nn.Module):
    def __init__(self, imgH=32, nc=1, nclass=12, nh=32, n_rnn=1, leakyRelu=False):
        super(SmallCRNN, self).__init__()

        ks = [3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1]
        ss = [1, 1, 1, 1, 1]
        nm = [16, 32, 64, 64, 64]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(f'conv{i}',
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            cnn.add_module(f'relu{i}',
                           nn.LeakyReLU(0.2, inplace=True) if leakyRelu else nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(2, 2))
        convRelu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(2, 2))
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling2', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(4)
        cnn.add_module('pooling3', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))

        self.cnn = cnn
        self.nclass = nclass

        # Since h=2, nIn becomes c * h
        nIn = nm[-1] * 2  # c * h
        self.rnn = nn.Sequential(
            BidirectionalLSTM(nIn=nIn, nHidden=nh, nOut=nh),
            BidirectionalLSTM(nIn=nh, nHidden=nh, nOut=self.nclass)
        )

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()

        conv = conv.view(b, c * h, w)
        conv = conv.permute(2, 0, 1)

        output = self.rnn(conv)
        output = F.log_softmax(output, dim=2)

        return output
