import torch
import torch.nn.utils.prune as prune
from tqdm import tqdm
import util

model = torch.load('model.pth')

print(model.keys())

# parameters_to_prune = (
#     # (model['fc1.weight'], 'weight'),
# )
#
# prune.global_unstructured(
#     parameters_to_prune,
#     pruning_method=prune.L1Unstructured,
#     amount=0.2,
# )
#
# print(
#     "Sparsity in rnn.weight: {:.2f}%".format(
#         100. * float(torch.sum(model.rnn.weight == 0))
#         / float(model.rnn.weight.nelement())
#     )
# )
# print(
#     "Sparsity in fc1.weight: {:.2f}%".format(
#         100. * float(torch.sum(model.fc1.weight == 0))
#         / float(model.fc1.weight.nelement())
#     )
# )
# print(
#     "Global sparsity: {:.2f}%".format(
#         100. * float(
#             torch.sum(model.rnn.weight == 0)
#             + torch.sum(model.fc1.weight == 0)
#         )
#         / float(
#             model.rnn.weight.nelement()
#             + model.fc1.weight.nelement()
#         )
#     )
# )