import torch
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_topk_accuracy(y_pred, y, k = 5):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k

# Next up is the training function. This is similar to all the previous notebooks, but with the addition 
# of the `scheduler` and calculating/returning top-k accuracy.

# The scheduler is updated by calling `scheduler.step()`. This should always be called **after** 
# `optimizer.step()` or else the first learning rate of the scheduler will be skipped. 

# Not all schedulers need to be called after each training batch, some are only called after each epoch. 
# In that case, the scheduler does not need to be passed to the `train` function and can be called in 
# the main training loop.
def train(model, iterator, optimizer, criterion, scheduler, device, k=5):
    
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_k = 0
    
    model.train()
    
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred, _ = model(x)
        
        loss = criterion(y_pred, y)
        
        acc_1, acc_k = calculate_topk_accuracy(y_pred, y, k=k)
        
        loss.backward()
        
        optimizer.step()
        
        scheduler.step()
        
        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_k += acc_k.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_k /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_k


# The evaluation function is also similar to previous notebooks, with the addition of the top-k accuracy.
# As the one cycle scheduler should only be called after each parameter update, it is not called 
# here as we do not update parameters whilst evaluating.
def evaluate(model, iterator, criterion, device, k=5):
    
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_k = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc_1, acc_k = calculate_topk_accuracy(y_pred, y, k=k)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_k += acc_k.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_k /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_k


# Next, a small helper function which tells us how long an epoch has taken.
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_predictions(model, iterator, device):

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)

    return images, labels, probs