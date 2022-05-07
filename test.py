"""
Test a model and compute detection mAP

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import argparse
import torchvision
from torch.utils.data import DataLoader

import pocket

from hicodet.hicodet import HICODet
from models import SpatiallyConditionedGraph as SCG
from utils import DataFactory, custom_collate, test

def main(args):
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False

    """num_anno = torch.tensor(HICODet(None, anno_file=os.path.join(
        args.data_root, 'train.json')).anno_interaction)
    rare = torch.nonzero(num_anno < 10).squeeze(1)
    non_rare = torch.nonzero(num_anno >= 10).squeeze(1)"""

    dataloader = DataLoader(
        dataset=DataFactory(
            #update dataset name
            name='hifo', 
            partition=args.partition,
            data_root=args.data_root,
            detection_root=args.detection_dir,
        ), collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=True
    )
    
    net = SCG(
        dataloader.dataset.dataset.object_to_verb, 49,
        num_iterations=args.num_iter,
        max_human=args.max_human,
        max_object=args.max_object,
        box_score_thresh=args.box_score_thresh,
        num_classes = 120,
        box_nms_thresh=args.box_nms_thresh,
        #fg_iou_thresh=0.55
    )
    epoch = 0
    if os.path.exists(args.model_path):
        print("Loading model from ", args.model_path)
        checkpoint = torch.load(args.model_path, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint["epoch"]
    elif len(args.model_path):
        print("\nWARNING: The given model path does not exist. "
            "Proceed to use a randomly initialised model.\n")

    net.cuda()
    timer = pocket.utils.HandyTimer(maxlen=1)
    
    with timer:
        test_ap = test(net, dataloader)
        #keep only the ap for new class
        test_ap = test_ap[600:]
    """print("Model at epoch: {} | time elapsed: {:.2f}s\n"
        "Full: {:.4f}, rare: {:.4f}, non-rare: {:.4f}".format(
        epoch, timer[0], test_ap.mean(),
        #test_ap[rare].mean(), test_ap[non_rare].mean()
    ))"""
    print(test_ap)
    print("Model at epoch: {} | time elapsed: {:.2f}s\n"
        "Full: {:.4f}".format(
        epoch, timer[0], test_ap.mean(),
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--data-root', default='hicodet', type=str)
    parser.add_argument('--detection-dir', default='hicodet/detections/hifo_data_preprocessing/base_nms-0.4_tl-2_best-base-update/test',
                        type=str, help="Directory where detection files are stored")
    parser.add_argument('--partition', default='test', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--max-human', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--model-path', default='', type=str)
    #parser.add_argument('--fusion-weight', default=1, type=float)
    parser.add_argument('--box-nms-thresh', default=0.5,type=float)
    
    args = parser.parse_args()
    print(args)

    main(args)
