import torch
import torch.nn as nn
import torch.utils.data.dataloader
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import util

# import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
df = pd.read_csv('pre_processed2.csv', index_col='time')

# instantiate the LSTM
model = util.LSTMModel(len(df.columns), 5)
model.to(device)

learning_rate = 0.001
n_epochs = 10000

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# model.train()

batch_size = 100
sec_len = 20
trn_dl = util.getDataSet("pre_processed2.csv", sec_len=sec_len, batch_size=batch_size, _norm=True)

counter = tqdm(range(1, n_epochs + 1))#, ncols=150)
counter.set_description(f'Loss: {1:.4f} | Acc: 0%')

# wandb.init(
#     project='AI-IOT',
#     config={
#         'learning_rate': learning_rate,
#         'n_epochs': n_epochs,
#         'batch_size': batch_size,
#         'sec_len': sec_len,
#         'model': 'LSTM',
#         'optimizer': 'SGD'
#         }
#     )

loss = None
correct = 0
last = []
for count, epoch in enumerate(counter):
    x_batch, y_batch = trn_dl.__iter__().__next__()

    out = model(x_batch.to(device))

    optimizer.zero_grad()

    loss = criterion(y_batch.to(device), out)
    accuracy = (torch.argmax(out.detach().cpu(), dim=1) == torch.argmax(y_batch.detach().cpu(), dim=1)).float().mean()

    loss.backward()
    optimizer.step()

    # for x, y in zip(out.detach().cpu().numpy(), y_batch.detach().cpu().numpy()):
    #     print(x, y)
    # exit()

    if epoch % 5 == 0:
        last.append(
            [
                *(y_batch.detach().cpu().numpy()[0]),
                *(out.detach().cpu().numpy()[0]),
                accuracy.item()*100,
                loss.item()
            ]
        )
        _temp = list(out.detach().cpu().numpy()[0])
        meant = int(torch.argmax(y_batch.detach().cpu(), dim=1)[0])
        counter.set_description(f'Loss: {loss.item():.4f} | Acc: {accuracy.item()*100:.0f}% | {_temp[0]:.4f},{_temp[1]:.4f},{meant}')

df = pd.DataFrame(last)
df.columns = ['Fake', 'Real', 'pFake', 'pReal', 'Accuracy', 'Loss']
df.to_csv('output.csv', index=False)
