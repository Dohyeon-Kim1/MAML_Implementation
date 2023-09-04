import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def conv_block_3x3(in_channel, out_channel):
  modules = [nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(2)]
  return nn.Sequential(*modules)

class MAML(nn.Module):

  def __init__(self, args):
    super().__init__()
    self.args = args
    self.loss_fn = nn.CrossEntropyLoss()
    
    self.conv1 = conv_block_3x3(3,32)
    self.conv2 = conv_block_3x3(32,32)
    self.conv3 = conv_block_3x3(32,32)
    self.conv4 = conv_block_3x3(32,32)
    self.fc = nn.Linear(800,self.args.num_classes)

  def functional_conv_block_forward(self, x, weights, biases, bn_weights, bn_biases, is_training):
    x = F.conv2d(x, weights, biases, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases, training=is_training)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
  return x

  def functional_forward(self, x, weights, is_training):
    for block in range(4):
      x = self.functional_conv_block_forward(x, weights[f'conv{block+1}.0.weight'], weights[f'conv{block+1}.0.bias'],
                                             weights.get(f'conv{block+1}.1.weight'), weights.get(f'conv{block+1}.1.bias'), is_training)
    x = x.view(x.size(0), -1)
    x = F.linear(x, weights['fc.weight'], weights['fc.bias'])
    return x
 
  def inner_loop(self, x_spt, y_spt):
    if not(self.args.train):
      self.args.num_updates = 10
    
    fast_weights = OrderedDict(self.named_parameters())
    for i in range(self.args.num_updates):
      spt_pred = self.functional_forward(x_spt, fast_weights, is_training=True)
      loss = self.loss_fn(spt_pred, y_spt)
      gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
      fast_weights = OrderedDict((name, param - self.args.inner_lr * grad) for ((name, param), grad) in zip(fast_weights.items(), gradients))
    return fast_weights

  def forward(self, x_spt, y_spt, x_qry, y_qry, is_training):
    fast_weights = self.inner_loop(x_spt, y_spt)
    qry_pred = self.functional_forward(x_qry, fast_weights, is_training)
    loss = self.loss_fn(qry_pred, y_qry)
    acc = (torch.argmax(qry_pred, dim=1) == y_qry).sum() / len(y_qry)
    return loss, acc
