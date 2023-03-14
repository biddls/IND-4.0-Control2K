import torch
from matplotlib import pyplot as plt
import util
import pandas as pd

# load data
df = pd.read_csv('pre_processed.csv', index_col='time')

model = util.LSTMModel(len(df.columns), 50, 3)
model.load_state_dict(torch.load("best.pth"))
model.eval()
trn_dl = util.getDataSet("pre_processed.csv", sec_len=10, batch_size=1)
counter = 0
correct = 0
for x_batch, y_batch in trn_dl:
    out = model(x_batch)
    counter += 1
    if out.argmax() == y_batch.argmax():
        correct += 1
    # print(f"The model chose: {int(out.argmax())}\nThe correct choice was: {int(y_batch.argmax())}")
    # for i, col in enumerate(x_batch.T):
    #     plt.plot(col, 'r' if i == y_batch.argmax() else 'b')
    # plt.show()
print(f"Accuracy: {100*(correct/counter):.0f}%")