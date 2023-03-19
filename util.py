import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pandas as pd
import random

from torch.autograd.grad_mode import F


# create LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, layer_dim):
        super(LSTMModel, self).__init__()
        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        self.lstm = nn.RNN(input_dim, 500, layer_dim, batch_first=True)
        self.lstm1 = nn.RNN(500, 1000, layer_dim, batch_first=True)
        self.lstm2 = nn.RNN(1000, 1000, layer_dim, batch_first=True)
        self.lstm3 = nn.RNN(1000, 500, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(500, 200)
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)

        # activation
        self.activation = torch.nn.LeakyReLU()

        # softmax
        self.softmax = torch.nn.Softmax(dim=1)

        # sigmoid
        self.sigmoid = torch.nn.Sigmoid()

        # flatten
        self.flatten = torch.nn.Flatten()

    # the size of input is [batch_size, seq_len(15), input_dim(75)]
    # the size of logits is [batch_size, num_class]
    def forward(self, _input):
        output, _ = self.lstm(_input)
        output, _ = self.lstm1(output)
        output = self.sigmoid(output)
        output, _ = self.lstm2(output)
        output, _ = self.lstm3(output)
        output = self.fc(output[:, -1])
        output = self.fc1(output)
        output = self.activation(output)
        output = self.sigmoid(output)
        output = self.fc2(output)
        output = self.activation(output)
        output = self.fc3(output)
        output = self.softmax(output)
        output = self.sigmoid(output)
        # print(output[0])
        return output


def cosine(t_max, eta_min: (int, float) = 0):
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2

    return scheduler


class IterDataset(data.IterableDataset):
    def __init__(self, file: str, _sec_len: int = 50, batch_size: int = 200):
        df = pd.read_csv(file, index_col='time')
        length = len(df)
        df = df.to_numpy()
        # columns = df.columns
        self.sec_len = _sec_len
        self.batch_size = batch_size

        def genData(_sec_len, batch_size) -> (torch.Tensor, torch.Tensor):
            for x in range(batch_size):
                if bool(random.getrandbits(1)):
                    # replace the column with noise
                    df_subset = np.random.normal(0, 1, (_sec_len, 14))
                    # df_subset[:, choice] = np.ones(_sec_len)*10

                    # convert to numpy array and then to tensor
                    df_subset = torch.from_numpy(df_subset)
                    # create the answer tensor
                    answer = [0, 1]
                    # return the subset of data and the answer
                    yield df_subset.float(), torch.tensor(answer).float()
                else:
                    # selects a subset of rows to analise
                    rows = random.randrange(length - _sec_len)
                    df_subset = df[rows:rows + _sec_len, :]

                    # convert to numpy array and then to tensor
                    df_subset = torch.from_numpy(df_subset)

                    # create the answer tensor
                    answer = [1, 0]

                    # return the subset of data and the answer
                    yield df_subset.float(), torch.tensor(answer).float()

        self.generator = genData

    def __iter__(self):
        return self.generator(self.sec_len, self.batch_size)


def getDataSet(file: str, sec_len: int = 50, batch_size: int = 200) -> data.DataLoader:
    return data.DataLoader(IterDataset(file, sec_len, batch_size), batch_size=batch_size)


if __name__ == '__main__':
    # from matplotlib import pyplot as plt
    # a = IterDataset('pre_processed.csv', 50)
    # a = a.generator(50, batch_size=1).__next__()
    # print(a[0].shape)
    # print(a[1].shape)
    # print(a[1].argmax())
    # for i, col in enumerate(a[0].T):
    #     plt.plot(col, 'r' if i == a[1].argmax() else 'b')
    # plt.show()


    # trying to check that im not going insane
    # load data
    df = pd.read_csv('pre_processed.csv', index_col='time')

    # instantiate the LSTM
    model = LSTMModel(len(df.columns), 10)
    model.eval()
    trn_dl = getDataSet("pre_processed.csv", sec_len=1, batch_size=1)
    last_in, _ = trn_dl.__iter__().__next__()
    last_out = model(last_in)
    for i in range(10):
        x_batch, y_batch = trn_dl.__iter__().__next__()
        out = model(x_batch)
        if torch.equal(last_in, x_batch):
            print("same in")
        last = x_batch
        print(repr(out))
