import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pandas as pd
import random
from torch.autograd import Variable


# create LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, layer_dim):
        super(LSTMModel, self).__init__()

        # Building your LSTM
        self.lstm = nn.RNN(input_dim, 100, layer_dim, batch_first=True)
        self.lstm1 = nn.RNN(100, 20, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(20, 10)
        self.fc1 = nn.Linear(10, 2)

        # softmax
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, _input):
        output, _ = self.lstm(_input)
        output, _ = self.lstm1(output)

        output = self.fc(output[:, -1])

        output = self.fc1(output)
        output = self.softmax(output)
        return output


def cosine(t_max, eta_min: (int, float) = 0):
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2

    return scheduler


class IterDataset(data.IterableDataset):
    def __init__(self, file: str, _sec_len: int = 50, batch_size: int = 200, _norm=False):
        df = pd.read_csv(file, index_col='time')
        length = len(df)
        columCount = len(df.columns)
        df = df.to_numpy()
        # columns = df.columns
        self.sec_len = _sec_len
        self.batch_size = batch_size
        self.norm = _norm

        def genData(_sec_len, batch_size, norm) -> (torch.Tensor, torch.Tensor):
            for x in range(batch_size):
                if bool(random.getrandbits(1)):
                    # print("RANDOM")
                    # replace the column with noise
                    df_subset = np.random.normal(0, 1, (_sec_len, columCount))
                    # df_subset[:, choice] = np.ones(_sec_len)*10

                    # convert to numpy array and then to tensor
                    df_subset = torch.from_numpy(df_subset)

                    # create the answer tensor
                    answer = [0, 1]
                    # return the subset of data and the answer
                    yield Variable(df_subset.float()), Variable(torch.tensor(answer).float())
                else:
                    # print("REAL")
                    # selects a subset of rows to analise
                    rows = random.randrange(length - _sec_len)
                    df_subset = df[rows:rows + _sec_len, :]

                    if norm:
                        # normalise the data
                        _std = df_subset.std(axis=0)
                        _std = ((_std == 0) * df_subset.mean(axis=0))+_std
                        df_subset = (df_subset - df_subset.mean(axis=0)) / _std

                    # convert to numpy array and then to tensor
                    df_subset = torch.from_numpy(df_subset)

                    # create the answer tensor
                    answer = [1, 0]

                    # return the subset of data and the answer
                    yield Variable(df_subset.float()), Variable(torch.tensor(answer).float())

        self.generator = genData

    def __iter__(self):
        return self.generator(self.sec_len, self.batch_size, self.norm)


def getDataSet(file: str, sec_len: int = 50, batch_size: int = 200, _norm=False) -> data.DataLoader:
    return data.DataLoader(IterDataset(file, sec_len, batch_size, _norm=_norm), batch_size=batch_size)


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
