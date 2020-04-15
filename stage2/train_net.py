import argparse
import os
import json
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.models as models
import torch.nn as nn
import random
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
#from torch.utils.tensorboard import SummaryWriter # not in provided enviroment
from stage2.stage2_utils import SvhnDatasetDigits, FocalLoss, MySmallNet
import time
#from sklearn.metrics import precision_recall_fscore_support # not in provided enviroment
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default = "stage2/config.json")
parser.add_argument("--logging", type=bool, default = False)





def get_config(path):
    with open(path, "r") as fp:
        config = json.load(fp)
    return config


def worker_func(rank, args):
    ngpu = torch.cuda.device_count()
    gpu = rank
    config = get_config(args.config_path)
    
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])
    if args.logging:
        writer = SummaryWriter(config['output_dir'])
    else:
        writer = None

    if ngpu > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["WORLD_SIZE"] = str(ngpu)
        dist.init_process_group("nccl", rank=rank, world_size=ngpu)

        
    # use pre trained vgg 16, extend output layer
    #model = models.vgg16(pretrained=True)
    #features = list(model.classifier.children())[:-1]
    #features.extend([nn.Linear(4096, 11)])
    #model.classifier = nn.Sequential(*features)
    
    model = MySmallNet()

    if ngpu > 0:
        
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        
        if ngpu > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [gpu])
            config['batch_size'] = int(config['batch_size'] / ngpu)
    
    loss_func = FocalLoss().cuda(rank)
    
    optimizer = torch.optim.SGD(model.parameters(), config['learning_rate'],
                            momentum=config['momentum'],
                            weight_decay=config['weight-decay'])
    
    trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomHorizontalFlip(p=1.0),
                                transforms.RandomVerticalFlip(p=1.0),
                                transforms.ColorJitter(.4,.4,.4),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_dataset = SvhnDatasetDigits("data/preprocessed/traindigits/train_reduced_negative.npy",
                                "data/preprocessed/traindigits/labels.json",
                                trans)
    val_dataset = SvhnDatasetDigits("data/preprocessed/testdigits/test_reduced_negative.npy",
                              "data/preprocessed/testdigits/labels.json",
                              trans)
    
    
    if ngpu > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size = config['batch_size'], shuffle = False, pin_memory = True,
        sampler = train_sampler)
    
    val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size = config['batch_size'], shuffle = False, pin_memory = True,
        sampler = val_sampler)
    
    
    for epoch in range(config['epochs']):
        print("Epoch: %d" % epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if epoch % config['eval_freq'] == 0:
            np.random.seed(1) # set seed
            evaluate(val_loader, model, epoch, rank, config, writer, loss_func)
            np.random.seed(int(time.time())) # unset seed
        if epoch % config['save_freq'] == 0:
            torch.save(model, os.path.join(config['output_dir'],"chkpoint_%d.pth" % epoch))
        train(train_loader, model, loss_func, optimizer, epoch, rank, config, writer)
        
        
    
    


    
def log_val_accuracy(gt, pred, writer, epoch, loss_func):
    """
    log validation stats with tensorboard SummaryWriter object
    """
    pred = torch.cat(pred, axis=0)
    gt = torch.cat(gt, axis=0)
    val_loss = loss_func(pred, gt)
    
    pred = torch.argmax(pred, axis=1).cpu().numpy()
    gt = torch.argmax(gt, axis=1).cpu().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(gt, pred, labels=range(11), average = "macro")
    
    pct_fp = np.sum(np.logical_and(gt == 10, pred != 10)) / len(gt)
    
    return float(val_loss), precision, recall, pct_fp
    
    
    

    

    
    

    
    
def train(loader, model, loss_func, optimizer, epoch, rank, config, writer):
    
    model.train()
    for ix, (imgs, target, _) in enumerate(loader):
        if ix % 20 == 0:
            print("Training: epoch %d, %f done" % (epoch, ix / len(loader)))
            #if dist.is_initialized():
            #    dist.barrier()
        
        if rank is not None:
            imgs = imgs.cuda(rank)
            target = target.cuda(rank)
        output = model(imgs)
        loss = loss_func(output, target)


        optimizer.zero_grad()
        #loss = loss.sum()
        if writer is not None:
            writer.add_scalar("Train/loss", loss, (epoch * len(loader)) + ix)
        loss.backward()
        optimizer.step()
        
        
    


def evaluate(loader, model, epoch, rank, config, writer, loss_func):
    outputs = []
    targets = []
    
    
    model.eval()
    with torch.no_grad():
        for ix, (imgs, target, lab) in enumerate(loader):
            if ix % 20 == 0:
                print("Validation Round: %f done" % round(ix / len(loader), 2))
            if rank is not None:
                imgs = imgs.cuda(rank)
                target = target.cuda(rank)
            output = model(imgs)
            outputs.append(output)
            targets.append(target)
            
        if writer is not None:
            stats = log_val_accuracy(targets, outputs, writer, epoch, loss_func)
            print(stats)
            loss, precision, recall, fp = stats
            writer.add_scalar("Validation/loss", loss, epoch)
            writer.add_scalars("Validation/stats", {'precision':precision,
                                                'recall':recall,
                                                'fp':fp}, epoch)

            
              
if __name__ == "__main__":
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        mp.spawn(worker_func, nprocs=ngpus, args = (args,))
    elif ngpus == 1:
        worker_func(0, args)
    else:
        worker_func(None, args)
    

