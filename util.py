import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pandas as pd
import random
from torch.autograd import Variable


# create LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super(LSTMModel, self).__init__()

        # Building RNN
        self.rnn = nn.RNN(input_dim, 50, 1, batch_first=True)

        # Readout layer
        self.fc1 = nn.Linear(50, input_dim)

        # softmax
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, _input):
        output, _ = self.rnn(_input)
        output = self.fc1(output[:, -1])
        output = self.softmax(output)
        return output

class IterDataset(data.IterableDataset):
    def __init__(self, file: str, _sec_len: int = 50, batch_size: int = 200, _norm=False):
        df = pd.read_csv(file, index_col='time')
        length = len(df)

        columCount = len(df.columns)
        # print(columCount)

        self.df = df.to_numpy()
        self.sec_len = _sec_len
        self.batch_size = batch_size
        self.norm = _norm

        def genData(_sec_len, batch_size, norm) -> (torch.Tensor, torch.Tensor):
            for x in range(batch_size):
                # print("REAL")
                # selects a subset of rows to analise
                rows = random.randrange(length - _sec_len)
                df_subset = self.df[rows:rows + _sec_len, :]

                if norm:
                    # normalise the data
                    _std = df_subset.std(axis=0)
                    _std = ((_std == 0) * df_subset.mean(axis=0))+_std
                    df_subset = (df_subset - df_subset.mean(axis=0)) / _std

                # print(f"{df_subset.shape=}")

                # pick random number between 0 and columCount
                random_number = random.randrange(columCount)

                df_subset[:, random_number] += np.random.normal(0, 10, _sec_len)

                # convert to numpy array and then to tensor
                df_subset = torch.from_numpy(df_subset)

                # create the answer tensor
                answer = [0] * columCount
                answer[random_number] = 1


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
