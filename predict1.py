import os
import sys
# os.system("pip install /tmp/wbf/ -f ./ --no-index")

os.system("mkdir /kaggle/working/maskdet")
os.system("cp -r /kaggle/input/yolov8/maskdet/* /kaggle/working/maskdet")
os.chdir("/kaggle/working/maskdet/")
os.system("python setup.py install")

sys.path.append("/kaggle/working/maskdet")
sys.path.append("/kaggle/input/yolov8")
import os
import numpy as np
import pandas as pd
import cv2

import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib
import json, itertools
from collections import defaultdict
import tifffile as tiff
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from wbf_masks import wbf_masks

tqdm.pandas()
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.data import load_inference_source
from ultralytics.yolo.utils import ops
from ultralytics.yolo.engine.results import Results
import torch.nn.functional as F

image_dir = "/kaggle/input/hubmap-hacking-the-human-vasculature/test"

def encode_binary_mask(mask: np.ndarray) -> t.Text:
    # check input mask --
    if mask.dtype != np.bool_:
        raise ValueError("encode_binary_mask expects a binary mask, received dtype == %s" % mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError("encode_binary_mask expects a 2d mask, received shape == %s" % mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]
    
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str


def preprocess(im):
    if not isinstance(im, torch.Tensor):
        same_shapes = all(x.shape == im[0].shape for x in im)
        transformed_im = [LetterBox(1280, auto=same_shapes, stride=32)(image=x) for x in im]

        im = np.stack(transformed_im)
        im = im[..., ::-1].transpose((0, 3, 1, 2))  
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im)

    img = im.to(device)
    img = img.half() if half else img.float() 
    img /= 255 
    return img


def get_mask(protos, masks_in, bboxes, shape, mask_thresh=0.5):
    c, mh, mw = protos.shape 
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw) 
    gain = min(mh / shape[0], mw / shape[1]) 
    pad = (mw - shape[1] * gain) / 2, (mh - shape[0] * gain) / 2 
    top, left = int(pad[1]), int(pad[0])  
    bottom, right = int(mh - pad[1]), int(mw - pad[0])
    masks = masks[:, top:bottom, left:right] 
    
    masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(bboxes[:, :, None], 4, 1) 
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None] 

    masks = masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    return masks.gt_(mask_thresh) 


def postprocess(preds, img, orig_img_shape, conf_thres, iou_thres, mask_thersh=0.5, max_det=400):
    p = ops.non_max_suppression(preds[0], 
                                conf_thres=conf_thres, 
                                iou_thres=iou_thres, 
                                agnostic=False, 
                                max_det=max_det, 
                                nc=num_classes,
                                classes=None
                                )
    proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  
    pred=p[0] # nms 结果
    
    masks_in = pred[:, :4] 
    if masks_in.shape[0] == 0:
        # 没有检测结果
        masks = None
        boxes = None
    else:
        pred[:, :4] = ops.scale_boxes(
            img.shape[2:],  
            pred[:, :4],  
            orig_img_shape # 512, 512
            ) 
        masks = get_mask(proto[0], pred[:, 6:], pred[:, :4], orig_img_shape, mask_thersh)  
        boxes=pred[:, :6] 

    return boxes,masks



def predict_image(im0,model, conf_t, iou_t, mask_t):
    h,w,c = im0.shape
    
    im = preprocess([im0]) # im -> (n, 3, h, w) 0or1
    preds = model(im, augment=False, visualize=False)
    boxes, masks = postprocess(preds, im, (h,w), conf_t, iou_t, mask_thersh=mask_t)
    
    if masks is not None:
        boxes = boxes.cpu().numpy()
        bboxes = boxes[:,:4]
        scores = boxes[:,4]

        pred_masks = masks
        pred_masks = pred_masks.data.cpu().numpy() 

    else:
        scores = []
        bboxes = []
        pred_masks = []

    return scores, bboxes, pred_masks




def ensemble_func(im0, yolo_models, conf_t, iou_t, mask_t, wmf_iou_t):
    for idx, yolo_model in enumerate(yolo_models):
        if idx == 0:
            scores, pred_boxes, pred_masks = predict_image(im0, yolo_model, conf_t, iou_t, mask_t)
            models = [0] * len(scores)  
        else:
            sc,pb,mks = predict_image(im0,yolo_model,conf_t, iou_t, mask_t)
            if len(sc) > 0:
                models += [idx] * len(sc)
                pred_masks = np.vstack([pred_masks, mks])
                scores = np.hstack([scores, sc])
                pred_boxes = np.vstack([pred_boxes, pb])

    sort_idx = np.argsort(-scores)
    pred_masks = pred_masks[sort_idx]
    pred_boxes = pred_boxes[sort_idx]
    models = np.array(models)[sort_idx]
    scores = scores[sort_idx]

    pred_masks, scores, _ = wbf_masks(pred_masks, 
                                    pred_boxes, 
                                    scores, 
                                    models, 
                                    iou_thr=wmf_iou_t, 
                                    soft_weight=np.sum(MODEL_WEIGHTS),
                                    model_weights=MODEL_WEIGHTS
                                    )      

    pred_masks = np.array(pred_masks)
    scores = np.array(scores)
    
    sort_idx = np.argsort(-scores)
    pred_masks = pred_masks[sort_idx]
    scores = scores[sort_idx]


    return scores, pred_masks


import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
half = True
imgsz = 1280  
image_size = 512   
num_classes = 1 

def load_weights(weight_path):
    model = AutoBackend(
        weight_path,
        device=device,
        fp16=half,
        fuse=True,
        verbose=False)
    model.eval()
    return model

weight_paths = [
    '/kaggle/input/yolov8/maskdetfold0/fold0.pt',
    '/kaggle/input/yolov8/maskdetfold0/fold2.pt',
    '/kaggle/input/yolov8/maskdetfold0/fold3.pt',
    '/kaggle/input/yolov8/maskdetfold0/fold4.pt',
]
yolo_models = []
for index,weight_path in enumerate(weight_paths):
    model = load_weights(weight_path)
    yolo_models.append(model)


MODEL_WEIGHTS = [1, 1, 0.3, 1]


images_list = os.listdir(image_dir)

conf_t = 0.00001 ####
iou_t = 0.6 ####
mask_t= 0.5
wmf_iou_t = 0.65

ids = []
heights = []
widths = []
prediction_strings = []
for image_name in tqdm(images_list,total=len(images_list)):
    image_path = f'{image_dir}/{image_name}'
    img = cv2.imread(image_path) # h, w, c
    image_id = image_name.split('.')[0]
    
    h, w, c = img.shape
    
    scores, pred_masks = ensemble_func(img, yolo_models, conf_t, iou_t, mask_t, wmf_iou_t)

    prediction_string = ''
    if len(pred_masks) > 0:

        for conf, mk in zip(scores, pred_masks):
            kernel = np.ones((4, 4), np.uint8)
            mk = cv2.dilate(mk.astype(np.uint8), kernel, iterations=1)
            
            binary_mask = mk.astype(np.bool)
            encoded = encode_binary_mask(binary_mask)

            prediction_string += f"0 {conf} {encoded.decode('utf-8')} "
            
    ids.append(image_name.split('.')[0])
    heights.append(h)
    widths.append(w)
    prediction_strings.append(prediction_string)


    
submission = pd.DataFrame()
submission['id'] = ids
submission['height'] = heights
submission['width'] = widths
submission['prediction_string'] = prediction_strings
submission = submission.set_index('id')
submission.to_csv("/kaggle/working/submission.csv")


subname = "submission.csv"
for ff in os.listdir("/kaggle/working"):
    if ff == subname:
        pass
    else:
        os.system(f"rm -rf /kaggle/working/{ff}")