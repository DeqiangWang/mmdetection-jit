import argparse

import cv2,os
import torch
import torch.onnx as torch_onnx
from mmdet.apis import inference_detector, init_detector, show_result,inference_data
from mmdet.models import build_backbone,build_neck,build_head
import mmcv

root = os.getcwd()
 
def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('--config',default='configs/mask_rcnn_r50_fpn_1x.py', help='test config file path')
    parser.add_argument('--checkpoint',default=None, help='checkpoint file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = mmcv.Config.fromfile( os.path.join(root,args.config))
    data = torch.randn(1,3,800,800) 
    
    with torch.no_grad():
        backbone = build_backbone(config.model.backbone) 
        neck = build_neck(config.model.neck)
        rpn_head = build_head(config.model.rpn_head)

    backbone.eval()
    neck.eval()
    rpn_head.eval()
   
    #torch.jit.save(rpn_head,'./rpn_head.pt')
    exit()
     
if __name__ == '__main__':
    main()
