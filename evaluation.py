import os
import argparse
from collections import namedtuple
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as pDDP
from torchsummary import summary
from torchvision.utils import save_image
import gin
from ds_load import myLoadDS, myTestDS
from utils import CTCLabelConverter, ModelEma
from cnv_model import OrigamiNet, ginM
from test import evaluate

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@gin.configurable
def test(opt, train_data_path, train_data_list, valid_data_path, valid_data_list, test_data_path, test_data_list, experiment_name, batch_size, workers):
    os.makedirs(f'./saved_models/{experiment_name}', exist_ok=True)
    parOptions = namedtuple('parOptions', ['DP', 'DDP', 'HVD'])
    parOptions.__new__.__defaults__ = (False,) * len(parOptions._fields)
    gin.parse_config_file(opt.gin)
    pO = parOptions(**{ginM('dist'):True})

    train_dataset = myLoadDS(flist=train_data_list, dpath=train_data_path)
    valid_dataset = myLoadDS(flist=valid_data_list, dpath=valid_data_path)
    alph = train_dataset.alph | valid_dataset.alph
    test_dataset = myTestDS(flist=test_data_list, dpath=test_data_path, alph=alph)
    print('Alphabet :',len(test_dataset.alph),test_dataset.alph)
    print('Dataset Size :',len(test_dataset.fns))
    print('-'*80)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=opt.world_size, rank=opt.rank)

    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size , pin_memory=True, 
                    num_workers = workers, sampler=test_sampler)

    model = OrigamiNet()
    checkpoint = torch.load('saved_models/rus_gin_test/best_norm_ED.pth')
    for key in list(checkpoint['model']):
        checkpoint['model'][key.replace('module.', '')] = checkpoint['model'].pop(key)

    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda(opt.rank)

    #manual debugging purposes
    #print("Beginning mem:", torch.cuda.memory_allocated(device)/1024/1024/1024)
    #model = model.to(device)
    #print("After model to device:", torch.cuda.memory_allocated(device)/1024/1024/1024)

    model = pDDP(model, device_ids=[opt.rank], output_device=opt.rank,find_unused_parameters=False)
    model_ema = ModelEma(model)
    model_ema._load_checkpoint('saved_models/rus_gin_test/best_norm_ED.pth')
    print("Model EMA loaded...")
    converter = CTCLabelConverter(test_dataset.ralph.values())
    print("Converter loaded with alphabet...")
    model.eval()
    with torch.no_grad():
        preds_list = evaluate(model_ema.ema, test_loader, converter) #originally model was model_ema.ema

    os.makedirs(f'./test_results/{experiment_name}', exist_ok=True)
    with open(f'./test_results/{experiment_name}/output.txt', 'w') as f:
        f.writelines(preds_list)
        f.close()

def gInit(opt):
    gin.parse_config_file(opt.gin)
    cudnn.benchmark = True

def rSeed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)

def launch_fn(rank, opt):
    print("inside launch_fn...")
    print("rank =", rank)
    gInit(opt)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(opt.port)
    dist.init_process_group("gloo", rank=rank, world_size=opt.num_gpu)
    rSeed(opt.manualSeed)
    torch.cuda.set_device(rank)
    opt.world_size = opt.num_gpu
    opt.rank = rank
    test(opt)
    dist.destroy_process_group()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gin', help='Gin config file')

    opt = parser.parse_args()

    gInit(opt)
    opt.manualSeed = ginM('manualSeed')
    opt.port = ginM('port')
    opt.num_gpu = torch.cuda.device_count()
    mp.spawn(launch_fn, args=(opt,), nprocs=opt.num_gpu, start_method='fork')