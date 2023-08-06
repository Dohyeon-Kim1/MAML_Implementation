import torch
import torch.nn as nn

def conv_block_3x3(in_channel, out_channel):
  modules = [nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(2)]
  return nn.Sequential(*modules)

class MAML(nn.Module):

  def __init__(self,args):
    super().__init__()
    self.grad = []
    self.num_class = args.num_class
    self.num_update = args.num_update
    self.inner_lr = args.inner_lr
    self.outer_lr = args.outer_lr
    
    self.conv1 = conv_block_3x3(3,32)
    self.conv2 = conv_block_3x3(32,32)
    self.conv3 = conv_block_3x3(32,32)
    self.conv4 = conv_block_3x3(32,32)
    self.fc = nn.Linear(800,self.num_class)

    self.loss_fn = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(list(self.parameters()), lr=self.outer_lr)

  def inner_loop(self, x_spt, y_spt):
    learner = deepcopy(self)
    opt_in = torch.optim.SGD(list(learner.parameters()), lr=self.inner_lr)

    for i in range(self.num_update):
      pred = learner(x_spt)
      loss = self.loss_fn(pred, y_spt)
      opt_in.zero_grad()
      loss.backward()
      opt_in.step()

    opt_in.zero_grad()
    return learner

  def outer_loop(self, learner, x_qry, y_qry, accum_grad=True):
    pred = learner(x_qry)
    loss = self.loss_fn(pred, y_qry)
    acc = (torch.argmax(pred, dim=1) == y_qry).sum() / len(y_qry)

    if accum_grad:
      grad = []
      loss.backward()
      for param in learner.parameters():
        grad.append(param.grad)
      self.grad.append(grad)

    return loss, acc

  def reset_grad(self):
    self.grad = []

  def update(self):
    self.optimizer.zero_grad()
    for i, param in enumerate(self.parameters()):
      param.grad = torch.zeros_like(param)
      num_grad = len(self.grad)
      for j in range(num_grad):
        param.grad += self.grad[j][i] / num_grad
    self.optimizer.step()
    self.reset_grad()

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = x.view(x.size(0),-1)
    x = self.fc(x)
    return x
