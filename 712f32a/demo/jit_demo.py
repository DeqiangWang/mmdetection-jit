import argparse

import cv2,os
import torch
import torch.onnx as torch_onnx
from mmdet.apis import inference_detector, init_detector, show_result,inference_data

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

    model = init_detector(
        os.path.join(root,args.config) , None, device=torch.device('cuda', args.device))
    
  
    img = cv2.imread(os.path.join(root,'demo/demo.jpg'))
    data = inference_data(model,img)
     
    with torch.no_grad():
        jitted = torch.jit.script(model(return_loss=False,**data))

        
    # exit()
 
    # for i in range(1,11):
     
    #     img = cv2.imread(os.path.join(root,'demo','{}.jpg'.format(i)))
         
    #     result = inference_detector(model, img)

      
    #     show_result(
    #         img, result, model.CLASSES, score_thr=args.score_thr, wait_time=0)


if __name__ == '__main__':
    main()
