from __future__ import print_function

import torch
import numpy as np
import wandb
import utils
from voc_dataset import VOCDataset


def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model):
    filename = 'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    print("saving model at ", filename)
    torch.save(model, filename)


def train(args, model, optimizer, scheduler=None, model_name='model'):
    wandb.login(key="67e677676d60bdb2e9b198f36f6b1f34f9c565de")
    run = wandb.init(
        name = model_name,
        reinit = True,
        project = "16824_hw1",
        config = args
    )
    
    train_loader = utils.get_data_loader(
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0

    for epoch in range(args.epochs):
        total_loss = 0

        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)

            optimizer.zero_grad()
            output = model(data)

            ##################################################################
            # TODO: Implement a suitable loss function for multi-label
            # classification. You are NOT allowed to use any pytorch built-in
            # functions. Remember to take care of underflows / overflows.
            # Function Inputs:
            #   - `output`: Outputs from the network
            #   - `target`: Ground truth labels, refer to voc_dataset.py
            #   - `wgt`: Weights (difficult or not), refer to voc_dataset.py
            # Function Outputs:
            #   - `output`: Computed loss, a single floating point number
            ##################################################################
            eps = 1e-10
            sigmoid = 1 / (1 + torch.exp(-output))
            sigmoid = torch.clamp(sigmoid, eps, 1-eps)
            loss = -torch.mean(target * torch.log(sigmoid) + (1-target) * torch.log(1-sigmoid))
            total_loss += float(loss.item())
            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################
            
            loss.backward()
            optimizer.step()
            
        #     if cnt % args.log_every == 0:
        #         writer.add_scalar("Loss/train", loss.item(), cnt)
        #         print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                
        #         # Log gradients
        #         for tag, value in model.named_parameters():
        #             if value.grad is not None:
        #                 writer.add_histogram(tag + "/grad", value.grad.cpu().numpy(), cnt)

        #     optimizer.step()
            
        #     # Validation iteration
        #     if cnt % args.val_every == 0:
        #         model.eval()
        #         ap, map = utils.eval_dataset_map(model, args.device, test_loader)
        #         print("map: ", map)
        #         writer.add_scalar("map", map, cnt)
        #         model.train()
            
        #     cnt += 1

        # if scheduler is not None:
        #     scheduler.step()
        #     writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)
        
        scheduler.step()

        total_loss = float(total_loss / len(train_loader))
        model.eval()
        ap, map = utils.eval_dataset_map(model, args.device, test_loader)
        model.train()

        wandb.log({
            "train_loss": total_loss,
            "mAP": map,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)
    
    run.finish()

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map
