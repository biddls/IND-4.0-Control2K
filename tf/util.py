import random
import numpy as np
import pandas as pd
from tensorflow import convert_to_tensor
from tensorflow import float32 as tf_float32


def IterDataset(file: bytes, STD: float, _sec_len: int = 50):
    file = file.decode(encoding="utf-8")
    df = pd.read_csv(file, index_col='time')
    length = len(df)

    columCount = len(df.columns)
    # print(columCount)

    df = df.to_numpy()
    sec_len = _sec_len

    def genData(_sec_len):
        _STD = STD
        while True:
            # print("REAL")
            # selects a subset of rows to analise
            rows = random.randrange(length - _sec_len)
            df_subset = df[rows:rows + _sec_len, :]

            # normalise the data
            _std = df_subset.std(axis=0)
            _std = ((_std == 0) * df_subset.mean(axis=0)) + _std
            df_subset = (df_subset - df_subset.mean(axis=0)) / _std

            # print(f"{df_subset.shape=}")

            # pick random number between 0 and columCount
            random_number = random.randrange(columCount)

            df_subset[:, random_number] += np.random.normal(0, _STD, _sec_len)

            # create the answer tensor
            answer = [0] * columCount
            answer[random_number] = 1

            df_subset = convert_to_tensor([df_subset], dtype=tf_float32)
            answer = convert_to_tensor([answer], dtype=tf_float32)

            received = yield df_subset, answer
            if received is not None:
                _STD = received

    return genData(sec_len)
