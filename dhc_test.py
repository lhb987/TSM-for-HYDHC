import os
import json
import time
import shutil

import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import itertools
from collections import OrderedDict

# TODO
from opts import parser
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool

best_prec1 = 0
num_class = 2
prefix = '{:09d}.jpg'

class ClassInfo:
    def __init__(self, class_, num_p, num_v, num_c):
        self.class_ = class_
        self.names = []
        self.num_p = num_p
        self.num_v = num_v
        self.num_c = num_c
classdist_test = [ClassInfo(0,0,0,0), ClassInfo(1,0,0,0), ClassInfo(2,0,0,0), ClassInfo(3,0,0,0)]

def main():
    global args, best_prec1, num_class, prefix
    args = parser.parse_args()
    test_list = gen_label(args.root_path, args.json_path)
    
    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local,
                dctidct=args.dct,
                )

    # crop_size = model.crop_size
    crop_size = 768
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()

    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    model.to(device)

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            print(device)
            checkpoint = torch.load(args.resume, map_location=device)

            ''' XXX
            Multi GPUs Model to Single device Model : Remove module. in front of key value 
            (https://developer0hye.tistory.com/entry/PyTorch-How-to-Switch-model-trained-on-Multi-GPUs-to-Single-GPU)
            (https://jangjy.tistory.com/323 : Another way)
            ''' 
            state_dict = checkpoint['state_dict']
            keys = state_dict.keys()
            values = state_dict.values()
            new_keys = []

            for key in keys:
                new_key = key[7:]    # remove the 'module.'
                new_keys.append(new_key)

            new_dict = OrderedDict(list(zip(new_keys, values)))
            model.load_state_dict(new_dict)

            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
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

    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    # TODO
    test_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, test_list, num_segments=args.num_segments,
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
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    tmp = [x.strip().split(' ') for x in open(test_list)]
    for obj in tmp:
        name = obj[0].split('/')[0]
        target = int(obj[2]) - 1
        if name not in classdist_test[target].names:
            classdist_test[target].names.append(name)
            classdist_test[target].num_p = classdist_test[target].num_p + 1
        classdist_test[target].num_v = classdist_test[target].num_v + 1
        classdist_test[target].num_c = classdist_test[target].num_c + (int(obj[1]) // 100)
    print('\n' + 'Test Distribution')
    for c in classdist_test:
        if not c.num_p == 0:
            # TODO
            output0 = ('Class : {:01d} - # of People : {:02d}, # of Videos : {:02d}, # of Clips : {}'.format(c.class_, c.num_p, c.num_v, c.num_c))
            print(output0)
    print()

    # TODO
    '''
    video = torchvision.io.VideoReader(args.video_path)
    validate(video_clip, model, criterion, 0)
    '''
    test(test_loader, model, criterion, device)

# TODO
def gen_label(root_path, json_path):
    id = 0
    output_test = []

    if os.path.exists(json_path):
        with open(json_path) as f1:
            data = json.load(f1)
            for img, annotation in zip(data['images'], data['annotations']):
                if id != img['id']:
                    id = img['id']
                    
                    file_name = img['file_name']
                    path = os.path.dirname(file_name)

                    nframes = img['nframes']

                    phq = annotation['phq9']
                    if phq < 5:
                        label = 1
                    # elif phq < 10:
                    #     label = 2
                    # elif phq < 20:
                    #     label = 3
                    else:
                        label = 2
                    
                    output_test.append('%s %d %d %d' % (path, nframes, label, 0))

    test_list = os.path.join(root_path + "/test_list.txt")
    print(test_list)
    with open(test_list, 'w') as f2:
        f2.write('\n'.join(output_test))

    return test_list


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

# TODO
def test(test_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    video_list = []
    participant = []
    with torch.no_grad():
        for i, (input, target, path) in enumerate(test_loader):
            target = target - 1
            target = target.to(device)
            input = input.to(device)

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
            prec1 = accuracy(output.data, target, topk=(1, 1))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0].item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # TODO
            '''
            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                        i, len(test_loader), batch_time=batch_time, loss=losses,
                        top1=top1))
                print(output)

    output1 = ('Testing Results: Prec@1 {top1.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, loss=losses))
    print(output1)
    '''

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
    
    for P in participant:
        output2 = 'Participant : {}, # of Pred : {:02d}'.format(P.name, P.number)
        print(output2)
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
        output3 = ('Class : {:d} | Pred_0 : {:02d}, Pred_1 : {:02d}'.format(P.target, pred_0, pred_1))
        print(output3 + '\n')

    return top1.avg

if __name__ == '__main__':
    main()