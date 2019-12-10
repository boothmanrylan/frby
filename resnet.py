import sys
import argparse

import torch
import torch.nn as nn
from torch import optim

import horovod.torch as hvd

from resnet_model import SimpleLines, ResidualNetwork

def gather(val, name):
    """ return average value of tensor from all nodes"""
    tensor = val.clone().detach()
    avg_tensor = hvd.allreduce(tensor, name=name)
    # if avg_tensor only has one element return it as a python number
    if torch.tensor(avg_tensor.shape).eq(1).all():
        avg_tensor = avg_tensor.item()
    return avg_tensor

def Print(message):
    """print a message from the root node only"""
    if not hvd.rank():
        print(message)

def evaluate(model, loader, sampler):
    with torch.no_grad():
        accuracy = 0.0
        for data, target in loader:
            output = model(data.cuda())
            prediction = output.argmax(dim=1, keepdim=True).view(-1)
            accuracy += prediction.eq(target.cuda().data).cpu().float().sum()

        accuracy /= len(sampler)
        accuracy = gather(accuracy, 'avg_accuracy') # avg accuracy over all GPUs
    return accuracy

hvd.init()
seed = 12345
torch.manual_seed(seed)
torch.cuda.set_device(hvd.local_rank())
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True

scratch = "/scratch/r/rhlozek/rylan/"

# height of 10944 is closest to what the Zhang et al. paper has
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="# of train epochs")
parser.add_argument("--height", type=int, default=512, help="data height")
parser.add_argument("--width", type=int, default=64, help="data width")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--train_N", type=int, default=10000, help="# of train samples")
parser.add_argument("--eval_N", type=int, default=1000, help="# of eval samples")
parser.add_argument("--scale", type=float, default=3, help="signal strength")
parser.add_argument("--lr", type=float, default=0.2, help="sgd learning rate")
parser.add_argument("--m", type=float, default=0.9, help="sgd momentum")
parser.add_argument("--wd", type=float, default=5e-4, help="sgd weight decay")
parser.add_argument("--path", type=str, default=scratch, help="model directory")
args = parser.parse_args()

train_dataset = SimpleLines(height=args.height, width=args.width,
                            scale=args.scale, N=args.train_N)
eval_dataset = SimpleLines(height=args.height, width=args.width,
                           scale=args.scale, N=args.eval_N)
val_dataset = SimpleLines(height=args.height, width=args.width,
                          scale=args.scale, N=16*hvd.size())

train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank()
)
eval_sampler = torch.utils.data.distributed.DistributedSampler(
    eval_dataset, num_replicas=hvd.size(), rank=hvd.rank()
)
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=hvd.size(), rank=hvd.rank()
)

train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler,
                                           batch_size=args.batch_size)
eval_loader = torch.utils.data.DataLoader(eval_dataset, sampler=eval_sampler,
                                          batch_size=args.batch_size)
val_loader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler,
                                         batch_size=16)

model = ResidualNetwork((args.height, args.width))
model.cuda() # move model to GPU

optimizer = hvd.DistributedOptimizer(
    torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.m,
                    weight_decay=args.wd),
    named_parameters=model.named_parameters()
)

# ensure all GPUs start with the same weights
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

criterion = nn.CrossEntropyLoss()

model.train() # place model in training mode
for epoch in range(args.epochs):
    for batch, (data, target) in enumerate(train_loader):
        optimizer.zero_grad() # clear previous gradients
        loss = criterion(model(data.cuda()), target.cuda())
        loss.backward() # run backpropagation
        optimizer.step() # update model parameters
        if batch % 2:
            acc = evaluate(model, val_loader, val_sampler)
            Print(f"Epoch: {epoch + 1}\tBatch: {batch + 1}\t"
                    f"Loss: {loss.item():.3}\tValidation Accuracy: {acc}")

# model.eval()
accuracy = evaluate(model, eval_loader, eval_sampler)

Print(f"Accuracy: {accuracy}")

if not hvd.rank():
    torch.save(model.state_dict(), args.path + 'model_dict')


