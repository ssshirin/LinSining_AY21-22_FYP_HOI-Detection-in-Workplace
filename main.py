"""
Train and validate with distributed data parallel

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import argparse
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

import pocket
from pocket.data import HICODet

from models import SpatiallyConditionedGraph as SCG
from utils import custom_collate, CustomisedDLE, DataFactory

def main(rank, args):

    dist.init_process_group(
        #windows use gloo
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    trainset = DataFactory(
        name=args.dataset, partition=args.partitions[0],
        data_root=args.data_root,
        detection_root=args.train_detection_dir,
        flip=True,
    )

    valset = DataFactory(
        name=args.dataset, partition=args.partitions[1],
        data_root=args.data_root,
        detection_root=args.val_detection_dir
    )

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank)
    )

    val_loader = DataLoader(
        dataset=valset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            valset, 
            num_replicas=args.world_size, 
            rank=rank)
    )

    # Fix random seed for model synchronisation
    torch.manual_seed(args.random_seed)

    if args.dataset == 'hicodet' or 'hifo':
        object_to_target = train_loader.dataset.dataset.object_to_verb
        human_idx = 49
        num_classes = 120
    elif args.dataset == 'vcoco':
        object_to_target = train_loader.dataset.dataset.object_to_action
        human_idx = 1
        num_classes = 24
    net = SCG(
        object_to_target, human_idx, num_classes=num_classes,
        num_iterations=args.num_iter, postprocess=False,
        max_human=args.max_human, max_object=args.max_object,
        box_score_thresh=args.box_score_thresh,
        distributed=True, 
        box_nms_thresh=args.box_nms_thresh,
       # fg_iou_thresh=0.55
        #fusion_weight
        #fusion_weight=args.fusion_weight,
    )
    
    KEEP = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116
    ]
    if os.path.exists(args.checkpoint_path):
        print("=> Rank {}: continue from saved checkpoint".format(
            rank), args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        #print(checkpoint['model_state_dict'])
        #print(checkpoint['model_state_dict']['interaction_head.box_pair_predictor.weight'].size())
        checkpoint['model_state_dict']['interaction_head.box_pair_predictor.weight'] = \
            checkpoint['model_state_dict']['interaction_head.box_pair_predictor.weight'][KEEP]
        checkpoint['model_state_dict']['interaction_head.box_pair_predictor.bias'] = \
            checkpoint['model_state_dict']['interaction_head.box_pair_predictor.bias'][KEEP]
        new_weight = torch.zeros(3,2048)
        new_bias = torch.zeros(3)
        checkpoint['model_state_dict']['interaction_head.box_pair_predictor.weight']= \
            torch.cat((checkpoint['model_state_dict']['interaction_head.box_pair_predictor.weight'],new_weight),0)
        checkpoint['model_state_dict']['interaction_head.box_pair_predictor.bias']= \
            torch.cat((checkpoint['model_state_dict']['interaction_head.box_pair_predictor.bias'],new_bias),0)
        #print(checkpoint['model_state_dict']['interaction_head.box_pair_predictor.weight'].size())
        #print(checkpoint['model_state_dict']['interaction_head.box_pair_predictor.bias'].size())
        #sys.exit()
        net.load_state_dict(checkpoint['model_state_dict'])
        #print(net)
        optim_state_dict = checkpoint['optim_state_dict']
        sched_state_dict = checkpoint['scheduler_state_dict']
        
        epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
    else:
        print("=> Rank {}: start from a randomly initialised model".format(rank))
        optim_state_dict = None
        sched_state_dict = None
        epoch = 0; iteration = 0

    engine = CustomisedDLE(
        net,
        train_loader,
        val_loader,
        num_classes=num_classes,
        print_interval=args.print_interval,
        cache_dir=args.cache_dir,
        #load state dict
        #optim_state_dict=optim_state_dict,
        #lr_sched_params=sched_state_dict
    )

    # Seperate backbone parameters from the rest
    param_group_1 = []
    param_group_2 = []
    param_group_3=[]
 
    for k, v in engine.fetch_state_key('net').named_parameters():
        if v.requires_grad:
            if k.startswith('module.backbone'):
                v.requires_grad = False
                param_group_1.append(v)
                #froze
            elif k.startswith('module.interaction_head.box_pair_head'):
                param_group_2.append(v)
            elif k.startswith('module.interaction_head.box_pair_suppressor') or k.startswith('module.interaction_head.box_pair_predictor'):
                param_group_3.append(v)
            else:
                raise KeyError(f"Unknown parameter name {k}")
    """print('group1:')
    print(param_group_1)
    print('group2:')
    print(param_group_2)
    print('group3:')
    print(param_group_3)
    sys.exit()"""
    
                
    # Fine-tune backbone with lower learning rate
    """optim = torch.optim.AdamW([
        {'params': param_group_1, 'lr': args.learning_rate * args.lr_decay},
        {'params': param_group_2}
        ], lr=args.learning_rate,
        weight_decay=args.weight_decay
    )"""
    #optim = torch.optim.AdamW(params=param_group_1, lr=args.learning.rate,weight_decay=args.weight_decay)
    #optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    optim = torch.optim.AdamW([
        {'params': param_group_1, 'lr': 0.},
         # Fine-tune interaction_head.box_head_pair with lower learning rate
        {'params': param_group_2, 
        #'lr': args.learning_rate * args.lr_decay
        },
        {'params': param_group_3,}
        ], lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    lambda1 = lambda epoch: 1. if epoch < args.milestones[0] else args.lr_decay
    lambda2 = lambda epoch: 1. if epoch < args.milestones[0] else args.lr_decay
    lambda3 = lambda epoch: 1. if epoch < args.milestones[0] else args.lr_decay
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=[lambda1, lambda2,lambda3]
    )
    # Override optimiser and learning rate scheduler
    engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
    engine.update_state_key(epoch=epoch, iteration=iteration)

    engine(args.num_epochs)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', required=True, type=int,
                        help="Number of subprocesses/GPUs to use")
    parser.add_argument('--dataset', default='hifo', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train', 'test'], type=str)
    parser.add_argument('--data-root', default='hicodet', type=str)
    parser.add_argument('--train-detection-dir', default='hicodet/detections/hifo_data_preprocessing/base_nms-0.4_tl-2_best-base-update/train', type=str)
    parser.add_argument('--val-detection-dir', default='hicodet/detections/hifo_data_preprocessing/base_nms-0.4_tl-2_best-base-update/test', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--num-epochs', default=16, type=int)
    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--learning-rate', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=4, type=int,
                        help="Batch size for each subprocess")
    parser.add_argument('--lr-decay', default=0.1, type=float,
                        help="The multiplier by which the learning rate is reduced")
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--max-human', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--milestones', nargs='+', default=[6,], type=int,
                        help="The epoch number when learning rate is reduced")
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--print-interval', default=100, type=int)
    parser.add_argument('--checkpoint-path', default='', type=str)
    parser.add_argument('--cache-dir', type=str, default='./checkpoints')
    parser.add_argument('--box-nms-thresh', default=0.4,type=float)
    #fusion_weight
    #parser.add_argument('--fusion-weight', default=1, type=float)

    args = parser.parse_args()
    print(args)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    mp.spawn(main, nprocs=args.world_size, args=(args,))
