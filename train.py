from __future__ import print_function, division

import torch.nn.parallel
import torch.optim
import time
import numpy as np

from test import test
from utils.utils import init_model_and_dataset, adjust_learning_rate, AverageMeter, accuracy
from utils.tb_visualizer import Logger


def train(ckpt, num_epochs, batch_size, device):
    start = time.time()

    num_workers = 0
    lr = 8e-4
    momentum = 0
    weight_decay = 0

    directory = 'data/'
    start_epoch = 0
    start_loss = 0
    print_freq = 10
    checkpoint_interval = 1
    evaluation_interval = 1

    logger = Logger('./logs')

    model, train_dataset, val_dataset, criterion_grid, optimizer = init_model_and_dataset(directory, device, lr,
                                                                                          weight_decay)

    # load the pretrained network
    if ckpt is not None:
        checkpoint = torch.load(ckpt)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_loss = checkpoint['loss']

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers, pin_memory=True)

    for epoch in range(start_epoch, num_epochs):
        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        batch_time = AverageMeter()
        total_time = AverageMeter()
        train_loss = AverageMeter()

        train_fingers_recall = AverageMeter()
        train_fingers_precision = AverageMeter()

        train_frets_recall = AverageMeter()
        train_frets_precision = AverageMeter()

        train_strings_recall = AverageMeter()
        train_strings_precision = AverageMeter()

        train_loss.update(start_loss)

        # switch to train mode
        model.train()

        for data_idx, data in enumerate(train_loader):
            batch_start = time.time()
            input = data['image'].float().to(device)
            target = data['fingers'].float().to(device)
            frets = data['frets'].float().to(device)
            strings = data['strings'].float().to(device)
            target_coord = data['finger_coord']
            frets_coord = data['fret_coord']
            strings_coord = data['string_coord']

            # compute output
            output = model(input)
            output1 = output[0].split(input.shape[0], dim=0)
            output2 = output[1].split(input.shape[0], dim=0)
            output3 = output[2].split(input.shape[0], dim=0)

            loss1 = sum(criterion_grid(o, target) for o in output1)
            loss2 = sum(criterion_grid(o, frets) for o in output2)
            loss3 = sum(criterion_grid(o, strings) for o in output3)

            loss = loss1/2 + loss2 + loss3
            print(f'loss1: {loss1}')
            print(f'loss2: {loss2}')
            print(f'loss3: {loss3}')

            # measure accuracy and record loss
            accuracy(output=output1[-1].data, target=target,
                     global_precision=train_fingers_precision, global_recall=train_fingers_recall, fingers=target_coord,
                     min_dist=10)

            accuracy(output=output2[-1].data, target=frets,
                     global_precision=train_frets_precision, global_recall=train_frets_recall,
                     fingers=frets_coord.unsqueeze(0), min_dist=5)

            accuracy(output=output3[-1].data, target=strings,
                     global_precision=train_strings_precision, global_recall=train_strings_recall,
                     fingers=strings_coord.unsqueeze(0), min_dist=5)

            train_loss.update(loss.item())

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - batch_start)
            total_time.update((time.time() - start)/60)

            if data_idx % print_freq == 0 and data_idx != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss.avg: {loss.avg:.4f}\t'
                      'Batch time: {batch_time:.4f} s\t'
                      'Total time: {total_time:.4f} min\n'
                      'FINGERS: \t'
                      'Recall(%): {top1:.3f}\t'
                      'Precision(%): {top2:.3f}\n'
                      'FRETS:   \t'
                      'Recall(%): {top6:.3f}\t'
                      'Precision(%): {top7:.3f}\n'
                      'STRINGS: \t'
                      'Recall(%): {top11:.3f}\t'
                      'Precision(%): {top12:.3f}\n'
                      '---------------------------------------------------------------------------------------------'
                    .format(
                    epoch, data_idx, len(train_loader), loss=train_loss, batch_time=batch_time.val,
                    total_time=total_time.val, top1=train_fingers_recall.avg * 100,
                    top2=train_fingers_precision.avg * 100, top6=train_frets_recall.avg * 100,
                    top7=train_frets_precision.avg * 100, top11=train_strings_recall.avg * 100,
                    top12=train_strings_precision.avg * 100))

        if epoch % evaluation_interval == 0:
            # evaluate on validation set
            print('---------------------------------------------------------------------------------------------\n'
                  'Train set:  ')

            t_recall1, t_recall2, t_recall3, t_precision1, t_precision2, t_precision3 = test(train_loader, model, device)
            print('Validation set:  ')
            e_recall1, e_recall2, e_recall3, e_precision1,  e_precision2,  e_precision3 = test(val_loader, model, device, show=False)

            print('---------------------------------------------------------------------------------------------\n'
                  '---------------------------------------------------------------------------------------------')

            # 1. Log scalar values (scalar summary)
            info = {'Train Loss': train_loss.avg,
                    '(Fingers) Train Recall': t_recall1, '(Fingers) Train Precision': t_precision1,
                    '(Fingers) Validation Recall': e_recall1, '(Fingers) Validation Precision': e_precision1,
                    '(Frets) Train Recall': t_recall2, '(Frets) Train Precision': t_precision2,
                    '(Frets) Validation Recall': e_recall2, '(Frets) Validation Precision': e_precision2,
                    '(Strings) Train Recall': t_recall3, '(Strings) Train Precision': t_precision3,
                    '(Strings) Validation Recall': e_recall3, '(Strings) Validation Precision': e_precision3}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch)

            # # 2. Log values and gradients of the parameters (histogram summary)
            # for tag, value in model.named_parameters():
            #     tag = tag.replace('.', '/')
            #     try: logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
            #     except ValueError: print('hey')
            #     logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)

            # 3. Log training images (image summary)
            info = {'images': input.view(-1, 300, 300).cpu().numpy()}

            for tag, images in info.items():
                logger.image_summary(tag, images, epoch)

        # remember best acc and save checkpoint
        if epoch % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss.avg
            }, "checkpoints/hg_ckpt_{0}.pth".format(epoch))


        print('Epoch time: {time}'.format(time=(time.time() - start_epoch)/60))