import os
import time
import string

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.distributed as dist
import numpy as np
import editdistance
from nltk.translate.bleu_score import sentence_bleu

from utils import Averager
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validation(model, criterion, evaluation_loader, converter, opt, parO):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    tot_ED = 0
    length_of_gt = 0
    bleu = 0.0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)

        text_for_loss, length_for_loss = converter.encode(labels)

        start_time = time.time()

        preds = model(image, '')
        forward_time = time.time() - start_time

        # Calculate evaluation loss for CTC deocder.
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        preds = preds.permute(1, 0, 2).log_softmax(2)  # to use CTCloss format

        # To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
        # https://github.com/jpuigcerver/PyLaia/issues/16
        torch.backends.cudnn.enabled = False
        cost = criterion(preds, text_for_loss, preds_size, length_for_loss).mean()
        torch.backends.cudnn.enabled = True

        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy.
        for pred, gt in zip(preds_str, labels):
            tmped = editdistance.eval(pred, gt)
            if pred == gt:
                n_correct += 1
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += tmped / float(len(gt))

            tot_ED += tmped
            length_of_gt += len(gt)
            bleu += sentence_bleu( [list(gt)], list(pred) ) 

    nelms = float(len(evaluation_loader.dataset))
    tot_ED = tot_ED / float(length_of_gt)
    val_loss = valid_loss_avg.val() if not parO.DDP else val_loss
    bleu /= nelms
    norm_ED /= nelms
    accuracy = n_correct / nelms * 100

    return val_loss, accuracy, norm_ED, tot_ED, bleu, preds_str, labels, infer_time