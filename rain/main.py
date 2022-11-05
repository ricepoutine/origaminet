import torch
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torch.optim as optim

import numpy as np
from tqdm import trange
import argparse
import time
import os
import sys
from collections import namedtuple
from cnv_model import OrigamiNet, ginM
from utils import CTCLabelConverter, Metric
from test import validation
import ds_load

import gin

parOptions = namedtuple('parOptions', ['DP', 'DDP', 'HVD'])
parOptions.__new__.__defaults__ = (False,) * len(parOptions._fields)

def WrkSeeder(_):
    return np.random.seed((torch.initial_seed())%(2**32))

def init_bn(model):
    if type(model) in [torch.nn.InstanceNorm2d, torch.nn.BatchNorm2d]:
        init.ones_(model.weight)
        init.zeros_(model.bias)
    elif type(model) in [torch.nn.Conv2d]:
        init.kaiming_uniform_(model.weight)

@gin.configurable
def train(opt, train_data_path, train_data_list, test_data_path, test_data_list, experiment_name, 
            train_batch_size, val_batch_size, workers, lr, valInterval, num_iter):

    ###NOTE this needs to change to work like origaminet's iam data handling
    ### WARNING: cringe incoming

    debug = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(f'./saved_models/{experiment_name}', exist_ok=True)

    train_dataset = ds_load.myLoadDS(train_data_list, train_data_path)
    valid_dataset = ds_load.myLoadDS(test_data_list, test_data_path , ralph=train_dataset.ralph)

    if debug:
        print('Alphabet :',len(train_dataset.alph),train_dataset.alph)
        for d in [train_dataset, valid_dataset]:
            print('Dataset Size :',len(d.fns))
            print('Max LbW : ',max(list(map(len,d.tlbls))) )
            print('#Chars : ',sum([len(x) for x in d.tlbls]))
            print('Sample label :',d.tlbls[-1])
            print("Dataset :", sorted(list(map(len,d.tlbls))) )
            print('-'*80)

    ###NOTE data loading and training
    train_loader  = torch.utils.data.DataLoader( train_dataset, batch_size=train_batch_size, shuffle=True, 
                pin_memory = True, num_workers = int(workers),
                sampler = None,
                worker_init_fn = WrkSeeder,
                collate_fn = ds_load.SameTrCollate
            )

    valid_loader  = torch.utils.data.DataLoader( valid_dataset, batch_size=val_batch_size , pin_memory=True, 
                    num_workers = int(workers), sampler=None)
    
    model = OrigamiNet()
    model.apply(init_bn)
    model.train()

    #if not pO.DDP, not gonna worry about that atm
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=10**(-1/90000))
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)
    converter = CTCLabelConverter(train_dataset.ralph.values())

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = 1e+6
    best_CER = 1e+6
    i = 0
    gAcc = 1
    epoch = 1
    titer = iter(train_loader)

    while(True):
        start_time = time.time()

        model.zero_grad()
        train_loss = Metric()

        for j in trange(valInterval, leave=False, desc='Training'):

            try:
                image_tensors, labels = next(titer)
            except StopIteration:
                epoch += 1
                titer = iter(train_loader)
                image_tensors, labels = next(titer)
                
            image = image_tensors.to(device)
            text, length = converter.encode(labels)
            batch_size = image.size(0)

            replay_batch = True
            maxR = 3
            while replay_batch and maxR>0:
                maxR -= 1
                
                preds = model(image,text).float()
                preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
                preds = preds.permute(1, 0, 2).log_softmax(2)
                
                if i==0:
                    print('Model inp : ',image.dtype,image.size())
                    print('CTC inp : ',preds.dtype,preds.size(),preds_size[0])

                # To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
                torch.backends.cudnn.enabled = False
                cost = criterion(preds, text.to(device), preds_size, length.to(device)).mean() / gAcc
                torch.backends.cudnn.enabled = True

                train_loss.update(cost)

                optimizer.zero_grad()
                default_optimizer_step = optimizer.step  # added for batch replay

                cost.backward()
                replay_batch = False
            
            if (i+1) % gAcc == 0:
                optimizer.step()
                model.zero_grad()

                if (i+1) % (gAcc*2) == 0:
                    lr_scheduler.step()
            
            i += 1
            
        # validation part
        if True:
            elapsed_time = time.time() - start_time
            start_time = time.time()

            model.eval()
            with torch.no_grad():
                valid_loss, current_accuracy, current_norm_ED, ted, bleu, preds, labels, infer_time = validation(model, criterion, valid_loader, converter, opt, pO)
        
            model.train()
            v_time = time.time() - start_time
              
            if current_norm_ED < best_norm_ED:
                best_norm_ED = current_norm_ED
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, f'./saved_models/{experiment_name}/best_norm_ED.pth')

                if ted < best_CER:
                    best_CER = ted
                
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy

                out  = f'[{i}] Loss: {train_loss.avg:0.5f} time: ({elapsed_time:0.1f},{v_time:0.1f})'
                out += f' vloss: {valid_loss:0.3f}'
                out += f' CER: {ted:0.4f} NER: {current_norm_ED:0.4f} lr: {lr_scheduler.get_lr()[0]:0.5f}'
                out += f' bAcc: {best_accuracy:0.1f}, bNER: {best_norm_ED:0.4f}, bCER: {best_CER:0.4f}, B: {bleu*100:0.2f}'
                print(out)

                with open(f'./saved_models/{experiment_name}/log_train.txt', 'a') as log: log.write(out + '\n')

        if i == num_iter:
            print('end the training')
            sys.exit()

def gInit(opt):
    gin.parse_config_file(opt.gin)
    parOptions(**{ginM('dist'):True})
    cudnn.benchmark = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin', help='Gin config file')
    opt = parser.parse_args()
    gInit(opt)
    opt.manualSeed = ginM('manualSeed')
    opt.port = ginM('port')

    opt.num_gpu = torch.cuda.device_count()

    train(opt)