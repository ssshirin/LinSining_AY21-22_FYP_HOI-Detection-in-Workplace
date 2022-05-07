"""
Fine-tune Faster R-CNN on HICO-DET

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

#from msilib.schema import Error
import os
import math
import json
import copy
import time
import cv2
from cv2 import transform
import torch
import bisect
import argparse
import torchvision
import numpy as np
from pocket.models.faster_rcnn import FastRCNNPredictor
import albumentations as A
import random
from tqdm import tqdm
from PIL import Image
from itertools import repeat, chain
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler

import pocket
from pocket.data import HICODet, DatasetConcat
from pocket.ops import RandomHorizontalFlip, to_tensor
from pocket.models import faster_rcnn, fasterrcnn_mobilenet_v3_large_fpn

class DetectorEngine(pocket.core.LearningEngine):
    def __init__(self, net, train_loader, val_loader, **kwargs):
        super().__init__(net, None, train_loader, **kwargs)
        self._val_loader = val_loader
        self.timer = pocket.utils.HandyTimer(1)

    def _on_each_iteration(self):
        #(self._state.inputs)
        #print(self._state.targets)
        self._state.output = self._state.net(self._state.inputs, targets=self._state.targets)
        self._state.loss = sum(loss for loss in self._state.output.values())
        self._state.optimizer.zero_grad()
        self._state.loss.backward()
        self._state.optimizer.step()

    def _on_end_epoch(self):
        with self.timer:
            ap, max_rec = self.validate()
            print(ap)
            ap = ap[80:]
            max_rec = max_rec[80:]
        print("\n=> Validation (+{:.2f})\n"
            "Epoch: {} | mAP: {:.4f}, mRec: {:.4f} | Time: {:.2f}s\n".format(
                time.time() - self._dawn, self._state.epoch,
                ap.mean().item(), max_rec.mean().item(), self.timer[0]
            ))
        print(ap)
        super()._on_end_epoch()

    @torch.no_grad()
    def validate(self, min_iou=0.5, nms_thresh=0.5):
        num_gt = torch.zeros(88)
        associate = pocket.utils.BoxAssociation(min_iou=min_iou)
        #update detection class from 80 to x, where x is the number of new classes
        meter = pocket.utils.DetectionAPMeter(
            88, algorithm='INT', nproc=10
        )
        self._state.net.eval()
        for batch in tqdm(self._val_loader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = self._state.net(inputs)
            assert len(output) == 1, "The batch size should be one"
            # Relocate back to cpu
            output = pocket.ops.relocate_to_cpu(output[0])
            target = batch[1][0]
            # Do NMS on ground truth boxes
            # NOTE This is because certain objects appear multiple times in
            # different pairs and different interactions
            keep_gt_idx = torchvision.ops.boxes.batched_nms(
                target['boxes'], torch.ones_like(target['labels']).float(),
                target['labels'], nms_thresh
            )
           
            gt_boxes = target['boxes'][keep_gt_idx].view(-1, 4)
            gt_classes = target['labels'][keep_gt_idx].view(-1)
            # Update the number of ground truth instances
            # Convert the object index to zero based
            for c in gt_classes:
                num_gt[c - 1] += 1
            # Associate detections with ground truth
            binary_labels = torch.zeros_like(output['scores'])
            unique_obj = output['labels'].unique()
            for obj_idx in unique_obj:
                det_idx = torch.nonzero(output['labels'] == obj_idx).squeeze(1)
                gt_idx = torch.nonzero(gt_classes == obj_idx).squeeze(1)
                if len(gt_idx) == 0:
                    continue
                binary_labels[det_idx] = associate(
                    gt_boxes[gt_idx].view(-1, 4),
                    output['boxes'][det_idx].view(-1, 4),
                    output['scores'][det_idx].view(-1)
                )
            meter.append(output['scores'], output['labels'] - 1, binary_labels)

        meter.num_gt = num_gt.tolist()
        ap = meter.eval()
        return ap, meter.max_rec

class HICODetObject(Dataset):
    def __init__(self, dataset, data_root, nms_thresh=0.5, random_flip=False):
        self.dataset = dataset
        self.nms_thresh = nms_thresh
        with open(os.path.join(data_root, 'coco80tohico80.json'), 'r') as f:
            corr = json.load(f)
        self.hico2coco91 = dict(zip(corr.values(), corr.keys()))
        #self.transform = RandomHorizontalFlip() if random_flip else None
        
        #use Albumentations for transform both image and bboxes
        #note that 1. Albumentation takes openCV image inputs, 2. the annotation used here is [x1, y1, x2, y2], which is in pascal_voc
        #now input openCV data for transform       
        self.transform = A.Compose([
            #A.SmallestMaxSize(max_size=475),
            #A.Sharpen(alpha=(0.4,0.7)),
            #A.ImageCompression(p=0.5),
            A.Perspective(scale=(0.08,0.1),p=0.5),
            A.HorizontalFlip(p=0.5)
            ],
            bbox_params=A.BboxParams(format='pascal_voc',label_fields=['category_ids'])) if random_flip else None
        
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        boxes = torch.cat([
            target['boxes_h'],
            target['boxes_o']
        ])
        # Convert ground truth boxes to zero-based index and the
        # representation from pixel indices to coordinates
        labels = torch.cat([
            49 * torch.ones_like(target['object']),
            target['object']
        ])
        # Convert HICODet object indices to COCO indices
        converted_labels = torch.tensor([int(self.hico2coco91[i.item()]) for i in labels])

        #convert PIL image to openCV (RGB)
        openCV_img = np.array(image)
        # Apply transform
        if self.transform is not None:
            #pocket default
            #image, boxes = self.transform(image, boxes)
            #albumentatiion
            transformed = self.transform(image=openCV_img, bboxes=boxes,category_ids=labels)
            #bboxes from numpy array to a Tensor
            openCV_img = transformed['image']
            boxes = transformed['bboxes']
            boxes = torch.Tensor(boxes)
            try:
                boxes[:, :2] -= 1
            except Exception:
                print(boxes)
            #openCV to PIL 
            image = Image.fromarray(openCV_img)

        image = to_tensor(image, input_format='pil')
        return [image], [dict(boxes=boxes, labels=converted_labels)]

def collate_fn(batch):
    images = []
    targets = []
    for im, tar in batch:
        images += im
        targets += tar
    return images, targets

"""
Batch sampler that groups images by aspect ratio
https://github.com/pytorch/vision/blob/master/references/detection/group_by_aspect_ratio.py
"""

def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)

class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with largest number
            # of elements
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size    

def _quantize(x, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized

def create_aspect_ratio_groups(aspect_ratios, k=0):
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    groups = _quantize(aspect_ratios, bins)
    # count number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    print("Using {} as bins for aspect ratio quantization".format(fbins))
    print("Count of instances per bin: {}".format(counts))
    return groups

def main(args):

    torch.cuda.set_device(0)
    torch.manual_seed(args.random_seed)

    train2015 = HICODetObject(HICODet(
        root=os.path.join(args.data_root, "hifo_data/images/train"),
        #root=os.path.join(args.data_root, "hico_20160224/images/train"),
        anno_file=os.path.join(args.data_root, "instances_train.json"),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    ), data_root=args.data_root, random_flip=True)
    test2015 = HICODetObject(HICODet(
        root=os.path.join(args.data_root, "hifo_data/images/test"),
        #root=os.path.join(args.data_root, "hico_20160224/images/test"),
        anno_file=os.path.join(args.data_root, "instances_test.json"),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    ), data_root=args.data_root)

    def div(a, b):
        return a / b
    use_train2015 = 'train' in args.training_data
    use_test2015 = 'test' in args.training_data
    if len(args.training_data) == 1 and use_train2015:
        trainset = train2015
        aspect_ratios = [div(*train2015.dataset.image_size(i)) for i in range(len(train2015))]
    elif len(args.training_data) == 1 and use_test2015:
        trainset = test2015
        aspect_ratios = [div(*test2015.dataset.image_size(i)) for i in range(len(test2015))]
    elif len(args.training_data) == 2 and use_train2015 and use_train2015:
        trainset = DatasetConcat(train2015, test2015)
        aspect_ratios = [
            div(*train2015.dataset.image_size(i)) for i in range(len(train2015))
        ] + [div(*test2015.dataset.image_size(i)) for i in range(len(test2015))]
    else:
        raise ValueError("Unknown dataset partition in ", args.training_data)

    sampler = torch.utils.data.RandomSampler(trainset)
    group_ids = create_aspect_ratio_groups(aspect_ratios, k=args.aspect_ratio_group_factor)
    batch_sampler = GroupedBatchSampler(sampler, group_ids, args.batch_size)
    train_loader = DataLoader(
        dataset=trainset,batch_sampler=batch_sampler,
        num_workers=4, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=test2015, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=collate_fn
    )
    if args.net == 'resnet':
        net = pocket.models.fasterrcnn_resnet_fpn('resnet50', pretrained=True,num_classes=89,trainable_backbone_layers=2,
        box_nms_thresh=0.4)
    elif args.net == 'mobilenet':
        net = pocket.models.fasterrcnn_mobilenet_v3_large_fpn('mobilenetv3', pretrained=True, num_classes=89,trainable_backbone_layers=2)
    net.cuda()
    
    engine = DetectorEngine(
        net, train_loader, val_loader,
        print_interval=args.print_interval,
        cache_dir=args.cache_dir,
        optim_params=dict(
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        ),
        lr_scheduler=True,
        lr_sched_params=dict(
            milestones=args.milestones,
            gamma=args.lr_decay
        )
    )

    engine(args.num_epochs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN on HIFO")
    parser.add_argument('--data-root', type=str, default='/home/students/s121md105_02/spatially-conditioned-graphs/hicodet')
    parser.add_argument('--training-data', nargs='+', default=['train'], type=str)
    parser.add_argument('--num-epochs', default=15, type=int)
    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--learning-rate', default=0.00025, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--milestones', nargs='+', default=[8, 12], type=int)
    parser.add_argument('--lr-decay', default=0.1, type=float)
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--print-interval', default=100, type=int)
    parser.add_argument('--cache-dir', type=str, default='./checkpoints')
    parser.add_argument('--net', type=str, default='resnet')

    args = parser.parse_args()
    print(args)

    main(args)
