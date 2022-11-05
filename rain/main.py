import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
from pathlib import Path
import argparse
import copy
from collections import namedtuple
import random
import time

from model import BasicBlock, Bottleneck, ResNet, LRFinder
from utils import count_parameters, train, evaluate, epoch_time, get_predictions
from plot import plot_images, plot_lr_finder, plot_confusion_matrix, plot_filtered_images, plot_filters
import ds_load

import gin

@gin.configurable
def train(opt, train_data_path, train_data_list, test_data_path, test_data_list, experiment_name, 
            train_batch_size, val_batch_size, workers, lr, valInterval, num_iter, wdbprj, continue_model=''):
    
    # We'll set the random seeds for reproducability.
    SEED = 42
    BATCH_SIZE = 200
    CPUS = 8
    EPOCHS = 4 
    N_IMAGES = 5
    N_FILTERS = 7

    learn_means_from_data = False
    show_sample_images = False
    print_model = False
    find_learning_rate = False

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    
    ###NOTE this needs to change to work like origaminet's iam data handling
    ### WARNING: cringe incoming

    debug = True

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
    train_loader  = torch.utils.data.DataLoader( train_dataset, batch_size=train_batch_size, shuffle=True if not HVD3P else False, 
                pin_memory = True, num_workers = int(workers),
                sampler = train_sampler if HVD3P else None,
                worker_init_fn = WrkSeeder,
                collate_fn = ds_load.SameTrCollate
            )

    valid_loader  = torch.utils.data.DataLoader( valid_dataset, batch_size=val_batch_size , pin_memory=True, 
                    num_workers = int(workers), sampler=valid_sampler if HVD3P else None)
    
    model = OrigamiNet()
    model.apply(init_bn)
    model.train()



    # image_dir = Path("/Volumes/Data/Work/Research/2022_10_ResNet/images")
    image_dir = Path.cwd() / "images"
    train_dir = image_dir / "train"
    test_dir = image_dir / "test"

    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]

    train_data = datasets.ImageFolder(root = train_dir, transform = transforms.ToTensor())

    if learn_means_from_data:
        means = torch.zeros(3)
        stds = torch.zeros(3)

        for img, label in train_data:
            means += torch.mean(img, dim = (1,2))
            stds += torch.std(img, dim = (1,2))

        means /= len(train_data)
        stds /= len(train_data)
        print(f'Calculated means: {means}')
        print(f'Calculated stds: {stds}')
    else:
        # these values are from the pretrained ResNet on 1000-class imagenet data
        means = [0.485, 0.456, 0.406]
        stds= [0.229, 0.224, 0.225]
    
    pretrained_size = 224
    train_transforms = transforms.Compose([
                            transforms.Resize(pretrained_size),
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomCrop(pretrained_size, padding = 10),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = means, std = stds)
                        ])

    test_transforms = transforms.Compose([
                            transforms.Resize(pretrained_size),
                            transforms.CenterCrop(pretrained_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = means, std = stds)
                        ])
    
    # We load our data with our transforms...
    train_data = datasets.ImageFolder(root = train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(root = test_dir, transform = test_transforms)

    VALID_RATIO = 0.9

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])
    
    # ...and then overwrite the validation transforms, making sure to 
    # do a `deepcopy` to stop this also changing the training data transforms.
    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transforms
    
    # To make sure nothing has messed up we'll print the number of examples 
    # in each of the data splits - ensuring they add up to the number of examples
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')
    
    # Next, we'll create the iterators with the largest batch size that fits on our GPU. 
    train_iterator = data.DataLoader(train_data, 
                        shuffle=True, 
                        batch_size=BATCH_SIZE, 
                        num_workers=CPUS, 
                        persistent_workers=True)
    valid_iterator = data.DataLoader(valid_data, 
                        batch_size=BATCH_SIZE, 
                        num_workers=CPUS, 
                        persistent_workers=True)
    test_iterator = data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=CPUS)
    
    print()
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of training iterations:   {len(train_iterator)}")
    print(f"Number of validation iterations: {len(valid_iterator)}")
    print(f"Number of test iterations:       {len(test_iterator)}")
    
    # To ensure the images have been processed correctly we can plot a few of them - 
    # ensuring we re-normalize the images so their colors look right.
    #if show_sample_images:
    #    N_IMAGES = 25
    #
    #    images, labels = zip(*[(image, label) for image, label in 
    #                            [train_data[i] for i in range(N_IMAGES)]])
    #
    #    classes = test_data.classes
    #    plot_images(images, labels, classes)
    
    # We will use a `namedtuple`to store: 
    #   the block class, 
    #   the number of blocks in each layer, 
    #   and the number of channels in each layer.
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

    resnet18_config = ResNetConfig(block = BasicBlock,
                               n_blocks = [2, 2, 2, 2],
                               channels = [64, 128, 256, 512])
    
    resnet34_config = ResNetConfig(block = BasicBlock,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512])

    # Below are the configurations for the ResNet50, ResNet101 and ResNet152 models. 
    # Similar to the ResNet18 and ResNet34 models, the `channels` do not change between configurations, 
    # just the number of blocks in each layer.
    resnet50_config = ResNetConfig(block = Bottleneck,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512])

    resnet101_config = ResNetConfig(block = Bottleneck,
                                    n_blocks = [3, 4, 23, 3],
                                    channels = [64, 128, 256, 512])

    resnet152_config = ResNetConfig(block = Bottleneck,
                                    n_blocks = [3, 8, 36, 3],
                                    channels = [64, 128, 256, 512])
    
    # The images in our dataset are 768x768 pixels in size. 
    # This means it's appropriate for us to use one of the standard ResNet models.
    # We'll choose ResNet50 as it seems to be the most commonly used ResNet variant. 

    # As we have a relatively small dataset - with a very small amount of examples per class - 40 images - 
    # we'll be using a pre-trained model.

    # Torchvision provides pre-trained models for all of the standard ResNet variants
    pretrained_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # We can see that the final linear layer for the classification, `fc`, has a 1000-dimensional 
    # output as it was pre-trained on the ImageNet dataset, which has 1000 classes.
    if print_model:
        print(pretrained_model)
    
    # Our dataset, however, only has 2 classes, so we first create a new linear layer with the required dimensions.
    IN_FEATURES = pretrained_model.fc.in_features 
    OUTPUT_DIM = len(test_data.classes)
    fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    
    # Then, we replace the pre-trained model's linear layer with our own, randomly initialized linear layer.
    # **Note:** even if our dataset had 1000 classes, the same as ImageNet, we would still remove the 
    # linear layer and replace it with a randomly initialized one as our classes are not equal to those of ImageNet.
    pretrained_model.fc = fc
    
    # The pre-trained ResNet model provided by torchvision does not provide an intermediate output, 
    # which we'd like to potentially use for analysis. We solve this by initializing our own ResNet50 
    # model and then copying the pre-trained parameters into our model.

    # We then initialize our ResNet50 model from the configuration...
    model = ResNet(resnet50_config, OUTPUT_DIM)
    
    # ...then we load the parameters (called `state_dict` in PyTorch) of the pre-trained model into our model.
    # This is also a good sanity check to ensure our ResNet model matches those used by torchvision.
    model.load_state_dict(pretrained_model.state_dict())

    print(f'The model has {count_parameters(model):,} trainable parameters')

    START_LR = 1e-7
    optimizer = optim.Adam(model.parameters(), lr=START_LR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('mps') #m1 core compat
    criterion = nn.CTCLoss(zero_infinity=True)
    model = model.to(device)
    criterion = criterion.to(device)
    
    # We then define our learning rate finder and run the range test.
    if find_learning_rate:
        END_LR = 10
        NUM_ITER = 100
        lr_finder = LRFinder(model, optimizer, criterion, device)
        lrs, losses = lr_finder.range_test(train_iterator, END_LR, NUM_ITER)
        
        # We can see that the loss reaches a minimum at around $3x10^{-3}$.
        # A good learning rate to choose here would be the middle of the steepest downward curve - which is around $1x10^{-3}$.
        plot_lr_finder(lrs, losses, skip_start = 30, skip_end = 30)
    
    # We can then set the learning rates of our model using discriminative fine-tuning - a technique 
    # used in transfer learning where later layers in a model have higher learning rates than earlier ones.

    # We use the learning rate found by the learning rate finder as the maximum learning rate - used in the final layer - 
    # whilst the remaining layers have a lower learning rate, gradually decreasing towards the input.
    
    FOUND_LR = 1e-3
    params = [
            {'params': model.conv1.parameters(), 'lr': FOUND_LR / 10},
            {'params': model.bn1.parameters(), 'lr': FOUND_LR / 10},
            {'params': model.layer1.parameters(), 'lr': FOUND_LR / 8},
            {'params': model.layer2.parameters(), 'lr': FOUND_LR / 6},
            {'params': model.layer3.parameters(), 'lr': FOUND_LR / 4},
            {'params': model.layer4.parameters(), 'lr': FOUND_LR / 2},
            {'params': model.fc.parameters()}
            ]

    optimizer = optim.Adam(params, lr = FOUND_LR)

    STEPS_PER_EPOCH = len(train_iterator)
    TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
    
    print(f"Starting training...")
    print(f"Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS} | Steps/Epoch: {STEPS_PER_EPOCH} | Total Steps: {TOTAL_STEPS}")

    MAX_LRS = [p['lr'] for p in optimizer.param_groups]

    scheduler = lr_scheduler.OneCycleLR(optimizer,
                                        max_lr = MAX_LRS,
                                        total_steps = TOTAL_STEPS)
    
    # Finally, we can train our model!
    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):
        
        start_time = time.monotonic()
        
        train_loss, train_acc_1, train_acc_5 = train(model, train_iterator, optimizer, criterion, scheduler, device, k=1)
        valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, valid_iterator, criterion, device, k=1)
            
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut5-model.pt')

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | ' \
            f'Train Acc @1: {train_acc_5*100:6.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | ' \
            f'Valid Acc @1: {valid_acc_5*100:6.2f}%')
        
        
    # Examine the test accuracies
    model.load_state_dict(torch.load('tut5-model.pt'))

    test_loss, test_acc_1, test_acc_k = evaluate(model, test_iterator, criterion, device, k=1)

    print(f'Test Loss: {test_loss:.3f} | Test Acc @1: {test_acc_1*100:6.2f}% | ' \
        f'Test Acc @1: {test_acc_k*100:6.2f}%')
    
    # ### Examining the Model
    # Get the predictions for each image in the test set...
    print()
    print("Getting predictions for images in the test set...")
    images, labels, probs = get_predictions(model, test_iterator, device)
    pred_labels = torch.argmax(probs, 1)
    
    # Plot the confusion matrix for the test results
    plot_confusion_matrix(labels, pred_labels, classes)
    
    # Show several images after they have been through the 'conv1' convolutional layer
    filters = model.conv1.weight.data
    il = [(image, label) for image, label in [train_data[i] for i in range(N_IMAGES)]]
    images, labels = zip(*il)
    plot_filtered_images(images, labels, classes, filters, n_filters=N_FILTERS)
    
    plot_filters(filters, title='After')
        
    return
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin', help='Gin config file')
    opt = parser.parse_args()
    opt.num_gpu = torch.cuda.device_count()

    train(opt)
