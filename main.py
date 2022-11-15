# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

''' XXX
Run .py file as .ipynb
'''
#%%
import os
import time
import shutil

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

''' XXX
Sort of Python Parser Module : 1. import argparse => Make parsing .py file separately and Import it (opts.py)
                               2. from optparse import OptionParser 
''' 
from opts import parser

''' XXX
Python Module Import Process : 1. Check sys.modules : Dictionary of imported Modules 
                               2. Check built-in modules : Python Standard Library
                               3. Check sys.path : Modules defined by User 
'''
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter

best_prec1 = 0

# XXX
class ClassInfo:
    def __init__(self, class_, num_p, num_v, num_c):
        self.class_ = class_
        self.names = []
        self.num_p = num_p
        self.num_v = num_v
        self.num_c = num_c
classdist_train = [ClassInfo(0,0,0,0), ClassInfo(1,0,0,0), ClassInfo(2,0,0,0), ClassInfo(3,0,0,0)]
classdist_val = [ClassInfo(0,0,0,0), ClassInfo(1,0,0,0), ClassInfo(2,0,0,0), ClassInfo(3,0,0,0)]

def main():
    global args, best_prec1
    ''' XXX
    For Jupyter Notebook
    ''' 
    # args = parser.parse_args()
    args = parser.parse_args(args=[])

    # return of dataset_config.return_dataset function
    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset, args.modality)

    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    ''' XXX
    1. Concatenating String with join Function
    2. Putting int Variable in the String with using %d and % symbols
    3. Putting int Variable in the String with format Function - It can be more formable ; Refer to format Function

    '''
    args.store_name = '_'.join(
        ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    args.store_name += '_merge_01_23_221115'
    print('storing name: ' + args.store_name)

    check_rootfolders()

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from), # XXX
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)

    # crop_size = model.crop_size
    crop_size = 768
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True) # XXX

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       GroupCenterCrop(crop_size),
                       GroupScale(int(scale_size)),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupCenterCrop(crop_size),
                       GroupScale(int(scale_size)),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    # XXX
    tmp = [x.strip().split(' ') for x in open(args.train_list)]
    for obj in tmp:
        name = obj[0].split('/')[0]
        target = int(obj[2]) - 1
        if name not in classdist_train[target].names:
            classdist_train[target].names.append(name)
            classdist_train[target].num_p = classdist_train[target].num_p + 1
        classdist_train[target].num_v = classdist_train[target].num_v + 1
        classdist_train[target].num_c = classdist_train[target].num_c + (int(obj[1]) // 100)
    print('\n' + 'Train Distribution')
    log_training.write('Train Distribution' + '\n')
    for c in classdist_train:
        if not c.num_p == 0:
            output0 = ('Class : {:01d} - # of People : {:02d}, # of Videos : {:02d}, # of Clips : {}'.format(c.class_, c.num_p, c.num_v, c.num_c))
            print(output0)
            log_training.write(output0 + '\n')
    print()
    log_training.write('\n')
    log_training.flush()

    tmp = [x.strip().split(' ') for x in open(args.val_list)]
    for obj in tmp:
        name = obj[0].split('/')[0]
        target = int(obj[2]) - 1
        if name not in classdist_val[target].names:
            classdist_val[target].names.append(name)
            classdist_val[target].num_p = classdist_val[target].num_p + 1
        classdist_val[target].num_v = classdist_val[target].num_v + 1
        classdist_val[target].num_c = classdist_val[target].num_c + (int(obj[1]) // 100)
    print('\n' + 'Test Distribution')
    log_training.write('Test Distribution' + '\n')
    for c in classdist_val:
        if not c.num_p == 0:
            output0 = ('Class : {:01d} - # of People : {:02d}, # of Videos : {:02d}, # of Clips : {}'.format(c.class_, c.num_p, c.num_v, c.num_c))
            print(output0)
            log_training.write(output0 + '\n')
    print()
    log_training.write('\n')
    log_training.flush()


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, epoch, log_training, tf_writer)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

            output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)

def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        ''' XXX
        Required Demension of Output & Target when using torch.nn.CrossEntropyLoss()
        1. Output : (batchsize x num_classes) - Not require passing the Softmax function
        2. Target : (batchsize) - Target has to be C-1 (C = num_classes)
        '''
        target = target - 1
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        ''' XXX
        if num_classes < 5, will error
        '''
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        prec1 = accuracy(output.data, target, topk=(1, 1))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0].item(), input.size(0))
        #top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))
            print(output)
            # XXX
            log.write(output + '\n')
            log.flush()

    # XXX
    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

# XXX
class Val_result:
    def __init__(self, path, logit, target, number):
        self.path = path
        self.sum_logit = logit
        self.target = target
        self.number = number

class Participant:
    def __init__(self, name, o_label, target, number):
        self.name = name
        self.o_label = []
        self.o_label.append(o_label)
        self.target = target
        self.number = number

    def o_label_update(self, o_label):
        self.o_label.append(o_label)

# XXX
def plot_confusion_matrix(con_mat, labels, title, cmap=plt.cm.get_cmap('Blues'), normalize=False):
    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks = np.arange(len(labels))
    nlabels = []
    for k in range(len(con_mat)):
        n = sum(con_mat[k])
        nlabel = '{0}(n={1})'.format(labels[k],n)
        nlabels.append(nlabel)
    plt.xticks(marks, labels)
    plt.yticks(marks, nlabels)

    thresh = con_mat.max() / 2.
    if normalize:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, '{0}%'.format(con_mat[i, j] * 100 / n), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    video_list = []
    participant = []
    with torch.no_grad():
        for i, (input, target, path) in enumerate(val_loader):
            target = target - 1
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # XXX
            for o, t, p in zip(output, target, path):
                append = 1
                for v in video_list:
                    if p == v.path:
                        v.sum_logit = v.sum_logit + o
                        v.number = v.number + 1
                        append = 0
                        break
                if append == 1:
                    t = t.item()
                    video_list.append(Val_result(p, o, t, 1))

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            prec1 = accuracy(output.data, target, topk=(1, 1))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0].item(), input.size(0))
            # top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output1 = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output1)

    # XXX
    labels = ['Normal', 'Mild', 'Serious', 'VerySerious']
    for R in video_list:
        name = R.path.split('/')[0]
        logit = R.sum_logit / R.number
        hypo = F.softmax(logit, dim=0)
        hypo = hypo.tolist()
        o_label = hypo.index(max(hypo))
        t = R.target
        append = 1
        for p in participant:
            if name == p.name:
                p.o_label_update(o_label)
                p.number = p.number + 1
                append = 0
                break
        if append == 1:
            participant.append(Participant(name, o_label, t, 1))   
    
    print()
    if log is not None:
        log.write('\n')
        log.flush()

    for P in participant:
        output2 = 'Participant : {}, # of Pred : {:02d}'.format(P.name, P.number)
        print(output2)
        # cm = confusion_matrix(R.target, R.pred)
        # plot_confusion_matrix(cm, labels=labels, title = 'Confusion Matrix of {}'.format(R.participant), normalize=False)
        pred_0, pred_1, pred_2, pred_3 = 0, 0, 0, 0
        for p in P.o_label:
            if p == 0:
                pred_0 = pred_0 + 1
            elif p == 1:
                pred_1 = pred_1 + 1
            elif p == 2:
                pred_2 = pred_2 + 1
            elif p == 3:
                pred_3 = pred_3 + 1
        # output3 = ('Class : {:d}, Pred_0 : {:02d}, Pred_1 : {:02d}, Pred_2 : {:02d}, Pred_3 : {:02d}'.format(P.target, pred_0, pred_1, pred_2, pred_3))
        output3 = ('Class : {:d}, Pred_0 : {:02d}, Pred_1 : {:02d}'.format(P.target, pred_0, pred_1))
        print(output3 + '\n')

        # XXX
        if log is not None:
            log.write(output2 + '\n')
            log.write(output3 + '\n' + '\n')
            log.flush()
    
    print()
    if log is not None:
        log.write(output1 + '\n')
        log.flush()
    
    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return top1.avg


def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    # XXX
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"  # Set the GPU 0 to use
    main()

# %%
