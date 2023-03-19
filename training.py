import torch
import torch.nn as nn
import torch.utils.data.dataloader
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
df = pd.read_csv('pre_processed.csv', index_col='time')


# instantiate the LSTM
model = util.LSTMModel(len(df.columns), 1)
model.to(device)


learning_rate = 1e-3
n_epochs = 100000
patience, trials = 100, 0
criterion = nn.CrossEntropyLoss()
# opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, amsgrad=False)
opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.01)
# scheduler = torch.optim.lr_scheduler.CyclicLR(opt, learning_rate, learning_rate*10,
# step_size_up=100, step_size_down=100, mode='triangular', gamma=1.0, cycle_momentum=False)

model.train()

losses = []
batch_size = 1000
sec_len = 50
trn_dl = util.getDataSet("pre_processed.csv", sec_len=sec_len, batch_size=batch_size)

counter = tqdm(range(1, n_epochs + 1), ncols=150)
counter.set_description(f'Loss: {0:.4f} Acc: {0:.2f}%')

loss = None
# x_batch = None
# y_batch = None
# for x_batch, y_batch in trn_dl:
#     x_batch = x_batch
#     y_batch = y_batch
correct = 0
for count, epoch in enumerate(counter):
    x_batch, y_batch = trn_dl.__iter__().__next__()
    opt.zero_grad()
    out = model(x_batch.to(device))
    loss = criterion(y_batch.to(device), out)
    loss.backward()
    opt.step()

    # scheduler.step()
    # for i, col in enumerate(x_batch.T):
    #     plt.plot(col)
    # plt.show()
    correct += torch.sum(out.argmax(dim=1) == y_batch.to(device).argmax(dim=1).int())

    # ouputs some logging to the progress bar and saves the loss to a list for graphing later
    if epoch % 5 == 0:
        counter.set_description(f'Loss: {loss.item():.4f} Acc: {100 * (correct / (count * batch_size)):.2f}%')
        losses.append(loss.item())
    # if loss.item() > best_acc:
    #     try:
    #         torch.save(model.state_dict(), 'best.pth')
    #     except RuntimeError as e:
    #         pass

plt.plot(range(0, len(losses)*5, 5), losses)
plt.axis(ymin=0, ymax=max(losses)*1.05)
plt.show()
