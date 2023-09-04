import torch
import numpy as np
from data_loader import MiniImagenet

def test(model, args):
  model.to(args.device)
  args.meta_batch_size = 1
  dataloader = MiniImagenet(args, "test")

  acc_list = []
  for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
    if step > 600:
      break
    
    x_spt, y_spt = x_spt.squeeze(0).to(args.device), y_spt.squeeze(0).to(args.device)
    x_qry, y_qry = x_qry.squeeze(0).to(args.device), y_qry.squeeze(0).to(args.device)

    _, acc = model(x_spt, y_spt, x_qry, y_qry, is_training=False)
    acc_list.append(acc.item())
  
  acc_list = np.array(acc_list)
  total_acc = np.mean(acc_list)
  ci95 = 1.96 * np.std(acc_list) / np.sqrt(600)
  
  return total_acc, ci95
