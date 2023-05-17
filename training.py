import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.dataloader
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import util
from util import LSTMModel

# import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
df = pd.read_csv('data/pre_processed2.csv', index_col='time')

# instantiate the LSTM
model: LSTMModel = util.LSTMModel(len(df.columns))
model.to(device)

learning_rate = 0.001
n_epochs = 100

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

batch_size = 100
sec_len = 50
trn_dl = util.getDataSet("data/pre_processed2.csv", sec_len=sec_len, batch_size=batch_size, _norm=True)

a = range(1, n_epochs + 1)
counter = tqdm(a)
counter.set_description(f'Loss: {1:.4f} | Acc: 000%')

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


for count, epoch in enumerate(a):
    x_batch, y_batch = trn_dl.__iter__().__next__()

    # plt.plot(x_batch[0].detach().cpu())
    # plt.show()
    # exit()

    out = model(x_batch.to(device))

    optimizer.zero_grad()

    loss = criterion(y_batch.to(device), out)

    loss.backward()
    optimizer.step()

    if epoch % (b := 50) == 0:
        accuracy = (torch.argmax(out.detach().cpu(), dim=1) == torch.argmax(y_batch.detach().cpu(), dim=1)).float().mean()
        counter.set_description(f'Loss: {loss.item():.4f} | Acc: {accuracy.item()*100:.0f}%')
        counter.update(b)

torch.save(model.state_dict(), 'model.pth')

x_batch, y_batch = trn_dl.__iter__().__next__()

model(x_batch[0].to(device).reshape(1, sec_len, -1))

# torch.onnx.export(model.to('cpu'), x_batch.to('cpu'), "pico/pico.onnx", verbose=True)
torch.onnx.export(model.to('cpu'),  # model being run
                  x_batch[0].to('cpu').reshape(1, sec_len, -1),  # model input (or a tuple for multiple inputs)
                  "pico/pico.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=12,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'])#,  # the model's output names
                  # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                  #               'output': {0: 'batch_size'}})
