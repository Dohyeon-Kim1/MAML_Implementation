import torch
import numpy as np
from data_loader import MiniImagenet

def test(model, args):
  model.to(args.device)
  args.meta_batch_size = 1
  dataloader = MiniImagenet(args, 'test')

  acc_list = []
  for step, (x_spt, y_spt, x_qry, y_qry, classes) in enumerate(dataloader):
    if step > 600:
      break
    x_spt, y_spt = x_spt.squeeze(0).to(args.device), y_spt.squeeze(0).to(args.device)
    x_qry, y_qry = x_qry.squeeze(0).to(args.device), y_qry.squeeze(0).to(args.device)

    learner = model.inner_loop(x_spt, y_spt)
    pred = learner(x_qry)
    acc = (torch.argmax(pred, dim=1) == y_qry).sum() / len(y_qry)
    acc_list.append(acc)
  
  acc_list = np.array(acc_list)
  acc = np.mean(acc_list)
  ci95 = 1.96 * np.std(acc_list) / np.sqrt(600)
  
  return acc, ci95
