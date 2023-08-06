import os
import torch
from data_loader import MiniImagenet

def train(model, args):
  Print_Iter = 100
  Save_Iter = 500
  print_loss, print_acc = 0.0, 0.0

  if not os.path.exists(f"{args.logdir}/{args.modeldir}/{args.seed}/"):
    os.makedirs(f"{args.logdir}/{args.modeldir}/{args.seed}/")

  model.to(args.device)
  dataloader = MiniImagenet(args, "train")

  for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
    x_spt, y_spt, x_qry, y_qry = x_spt.to(args.device), y_spt.to(args.device), x_qry.to(args.device), y_qry.to(args.device)
    task_loss, task_acc = [], []

    for i in range(args.meta_batch_size):
      learner = model.inner_loop(x_spt[i], y_spt[i])
      loss, acc = model.outer_loop(learner, x_qry[i], y_qry[i], accum_grad=True)

      task_loss.append(loss)
      task_acc.append(acc)

    model.update()

    train_loss = torch.stack(task_loss).mean().item()
    train_acc = torch.stack(task_acc).mean().item()

    if step != 0 and step % Print_Iter == 0:
      print(f"Step {step}/{args.meta_iterations}\tLoss:{round(train_loss,4)}\tAccuracy: {round(train_acc,4)}")
      print_loss, print_acc = 0.0, 0.0
    else:
      print_loss += train_loss / Print_Iter
      print_acc += train_acc / Print_Iter

    if args.save and step != 0 and step % Save_Iter == 0:
      torch.save(model.state_dict(), f"{args.logdir}/{args.modeldir}/{args.seed}/model{step}")
