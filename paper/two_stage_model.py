from skimage.utils.shape import view_as_blocks as block

import torch
import torch.nn as nn

def two_stage_model(modelA, modelB, dataloader):
    predictions = []
    targets = []
    for (data, target) in data_loader:
        data = data.cuda()
        d = D.reshape((D.shape[0] * D.shape[1],) + D.shape[2:])
        outputA = modelA(d).argmax(dim=1, keepdim=True)
        outputA = outputA.reshape((1, 1, D.shape[0], D.shape[1]))
        outputB = modelB(outputA.float().cuda())
        predictions.append(outputB.argmax(dim=1, keepdim=True))
        targets.append(target)
    return predictions, targets





