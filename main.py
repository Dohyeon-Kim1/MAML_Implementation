import argparse
import numpy as np
import random
import torch

from maml import MAML
from train import train
from test import test

parser = argparse.ArgumentParser(description='MAML_MiniImagenet')
parser.add_argument('--datasource', default='miniimagenet', type=str,
                    help='miniimagenet')
parser.add_argument('--num_classes', default=5, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--test_epoch', default=-1, type=int, 
                    help='test epoch, only work when test start')
parser.add_argument('--device', default='cuda', type=str,
                     help='used device when train or test')
parser.add_argument('--seed', type=int,
                    help='weight initialization, dataloader, random seed.')

## Training options
parser.add_argument('--metatrain_iterations', default=20000, type=int,
                    help='number of metatraining iterations.')
parser.add_argument('--num_updates', default=5, type=int,
                    help='number of updates for inner gradient updates')
parser.add_argument('--inner_lr', default=0.01, type=float, 
                    help='the learning rate of learner (inner-loop)')
parser.add_argument('--outer_lr', default=0.001, type=float, 
                    help='the learning rate of meta-learner (outer-loop)')
parser.add_argument('--meta_batch_size', default=4, type=int, 
                    help='number of tasks sampled per meta-update')
parser.add_argument('--update_batch_size', default=5, type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_batch_size_eval', default=15, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')
parser.add_argument('--save', default=1, type=int,
                    help='save checkpoint when training.')

## Model options
parser.add_argument('--num_filters', default=32, type=int,
                    help='number of filters for conv nets (32 for miniimagenet)')
parser.add_argument('--weight_decay', default=0.0, type=float, 
                    help='weight decay')

## Logging, saving, and testing options
parser.add_argument('--logdir', default='xxx', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--datadir', default='xxx', type=str, 
                    help='directory for datasets.')
parser.add_argument('--modeldir', type=str,
                    help='directory for each model')
parser.add_argument('--resume', default=0, type=int, 
                    help='resume training if there is a model available')
parser.add_argument('--train', default=1, type=int, 
                    help='True to train, False to test.')
parser.add_argument('--trial', default=0, type=int, 
                    help='trail for each layer')
parser.add_argument('--ratio', default=0.2, type=float, 
                    help='the ratio of meta-training tasks')

if __name__ == "__main__":
  args = parser.parse_args()
  print(args)

  if args.train == 1:
    random.seed(args.seed)
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)
    torch.cuda.manual_seed(args.seed+2)
  else:
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    torch.cuda.manual_seed(3)

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  model = MAML(args)

  if args.resume == 1 and args.train == 1:
    model_file = f"{args.logdir}/{args.modeldir}/{args.seed}/model{args.test_epoch}"
    print(model_file)
    model.load_state_dict(torch.load(model_file))

  if args.train == 1:
    model.train()
    train(model, args)
  else:
    model_file = f"{args.logdir}/{args.modeldir}/{args.seed}/model{args.test_epoch}"
    model.load_state_dict(torch.load(model_file))
    model.eval()
    acc, ci95 = test(model,args)
    print(f"Accuracy: {acc}, CI95 {ci95}")
