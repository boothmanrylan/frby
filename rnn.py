import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch_dataset import SimpleLines, RNNDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import horovod.torch as hvd

class RNN(nn.Module):
    def __init__(self, ninputs, nfeatures, nclasses, nlayers):
        super().__init__()
        self.lstm  = nn.LSTM(input_size=ninputs, hidden_size=nfeatures,
                             num_layers=nlayers)
        self.linear = nn.Linear(nfeatures, nclasses)

    def forward(self, input):
        x, _ = self.lstm(input)
        x = self.linear(x.mean(0))
        return x

class WaterFallCell(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(self.input_size, hidden_size)
        self.hx = torch.randn(batch_size, hidden_size).cuda()
        self.cx = torch.randn(batch_size, hidden_size).cuda()

    def forward(self, input):
        x = input[:, :self.input_size]
        self.hx, self.cx = self.cell(x, (self.hx, self.cx))
        x = torch.cat([self.hx.detach(), input[:, self.input_size:]], 1)
        return x

class WaterFall(nn.Module):
    def __init__(self, nclasses, ninputs, ncells, batch_size):
        super().__init__()
        assert not ninputs % ncells
        self.nclasses = nclasses
        cells = [WaterFallCell(input_size=(ninputs // ncells) + x,
                               hidden_size=(x + 1),
                               batch_size=batch_size)
                 for x in range(ncells)]
        self.model = nn.Sequential(*cells + [nn.Linear(ncells, self.nclasses)])

    def forward(self, input):
        output = torch.zeros((input.shape[0], input.shape[1], self.nclasses)).cuda()
        for i in range(input.shape[0]):
            output[i] = self.model(input[i])
        return output.mean(0)

def gather(val, name):
    tensor = val.clone().detach()
    avg_tensor = hvd.allreduce(tensor, name=name)
    # check if single element in avg_tensor
    if torch.tensor(avg_tensor.shape).eq(1).all():
        avg_tensor = avg_tensor.item()
    return avg_tensor

hvd.init()
torch.manual_seed(12345)
torch.cuda.set_device(hvd.local_rank())
torch.cuda.manual_seed(12345)
cudnn.benchmark = True

input_size = 2
batch = 64
rnn = WaterFall(nclasses=2, ninputs=input_size, ncells=input_size, batch_size=batch)
if not hvd.rank(): print(rnn)

rnn.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = hvd.DistributedOptimizer(
                torch.optim.Adam(
                    rnn.parameters(),
                    lr=0.0001,
                    weight_decay=0.0),
                named_parameters=rnn.named_parameters()
            )

hvd.broadcast_parameters(rnn.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

dataset = SimpleLines(height=input_size, width=32, scale=1e6, N=10000, add_channels=False)
sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
loader = DataLoader(dataset, sampler=sampler, batch_size=batch, num_workers=0,
        drop_last=True)

rnn.train()
for epoch in range(50):
    for batch, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        out = rnn(data.transpose(0, 1).cuda())
        loss = criterion(out, target.cuda())
        loss.backward()
        # gradient clipping to prevent vanishing/exploding gradients
        # helpful with lots of recurring layers
        nn.utils.clip_grad_norm_(rnn.parameters(), 2)
        optimizer.step()
    if not hvd.rank():
        print(f"Epoch: {epoch+1}\tLoss: {loss.item()}")

rnn.eval()
with torch.no_grad():
    accuracy = 0.0
    for batch, (data, target) in enumerate(loader):
        data = data.transpose(0, 1)
        out = rnn(data.cuda())
        pred = torch.sigmoid(out).round() # F.softmax(output, dim=-1).data.cpu().numpy()
        pred = out.argmax(dim=1, keepdim=True)
        accuracy += pred.eq(target.cuda().data.view_as(pred)).cpu().float().sum()
    accuracy /= len(sampler)
    accuracy = gather(accuracy, 'avg_accuracy')
    if not hvd.rank():
        print(f"Accuracy: {accuracy}")
