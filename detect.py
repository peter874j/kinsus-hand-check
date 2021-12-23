import argparse
import time
import os
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

class ObjectPoint(object):
    """Creates a point on a coordinate plane with values x and y.
        (for tensor)
    """    

    def __init__(self, boxList):
        '''Defines x and y variables'''
        self.x1 = int(boxList[0].item())
        self.y1 = int(boxList[1].item())
        self.x2 = int(boxList[2].item())
        self.y2 = int(boxList[3].item())


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://'))

    ### Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    ### Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    ### Read cfg file name
    cfgFileName = r'./config.yml'
    caliArray = np.load(r'./R3V6F_720p_matrix.npz')

    ### Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    ### Second-stage classifier
    '''
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    '''
    ### Set Dataloader
    vid_writer = None, None
    if webcam:
        view_img = True
        save_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, caliArray, img_size=imgsz)
    else:
        save_img = True
        ### 無加上畸變校正
        dataset = LoadImages(source, img_size=imgsz)

    ### Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    ### Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        oriImg = im0s.copy()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        ### Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        ### Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        ### Apply Classifier
        '''
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        '''
        ### Declare PCBBoxList, glovesBoxesList
        PCBBoxList, glovesBoxesList = [], []
        PCBBoxConf = 0.
        ### Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir)  if webcam else str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                ### Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                ### Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                ### Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    ### PCB Class
                    if cls == 0:
                        PCBBoxList = xyxy if conf >= PCBBoxConf else PCBBoxList
                        PCBBoxConf = conf if conf >= PCBBoxConf else PCBBoxConf
                    ### Gloves Class
                    elif cls == 1:
                        glovesBoxesList.append(xyxy)
            ### Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            ### save image or video results
            vid_writer = save_result(im0, save_path, vid_cap, vid_writer, dataset.mode, view_img, save_img)
        ### per image PCB & Gloves Box
        touchFlag, resultMask = judge_touching(oriImg, PCBBoxList, glovesBoxesList)   
        ### classfy results by folder
        OKFolderPath = r"./classify_results/OK"
        NGFolderPath = r"./classify_results/NG"
        if touchFlag:
            cv2.imwrite(os.path.join(NGFolderPath, os.path.basename(save_path)), im0s)
        else:
            cv2.imwrite(os.path.join(OKFolderPath, os.path.basename(save_path)), im0s)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

def judge_touching(img, PCBBoxList, glovesBoxesList, pixelThres=30):
    """

    Args:
        img (img): Scenes Image
        PCBBoxList (list): [x1, y1, x2, y2]
        glovesBoxesList (list): [[x1, y1, x2, y2], [x1, y1, x2, y2], ..., [x1, y1, x2, y2]]
        pixelThres (int, optional): Threshold of the Touching Distance. Defaults to 30.

    Returns:
        [bool]: touchFlag
        [img]: intersectionMask
    """    
    ### Initial Parameter
    touchFlag = False
    (imgH, imgW, ch) = img.shape
    glovesImg = np.zeros((imgH, imgW, ch), np.uint8)
    ### Mask of PCB, Gloves
    PCBMask = np.zeros((imgH, imgW), np.uint8)
    glovesMask = np.zeros((imgH, imgW), np.uint8)
    resultMask = np.zeros((imgH, imgW), np.uint8)
    intersectionMask = np.zeros((imgH, imgW), np.uint8)
    ### Point of PCB, GlovesL, GlovesR
    PCBBox = ObjectPoint(PCBBoxList)
    PCBMask[PCBBox.y1 + pixelThres:PCBBox.y2 - pixelThres, PCBBox.x1 + pixelThres:PCBBox.x2 - pixelThres] = 255

    ### Mask of Gloves
    lowerGloves = np.array([210, 210, 210])
    upperGloves = np.array([255, 255, 255]) 
    for glovesBoxList in glovesBoxesList:
        glovesBox = ObjectPoint(glovesBoxList)
        ### Region of Gloves
        glovesImg[glovesBox.y1:glovesBox.y2, glovesBox.x1:glovesBox.x2] = img[glovesBox.y1:glovesBox.y2, glovesBox.x1:glovesBox.x2]
        
    glovesMask = cv2.inRange(glovesImg, lowerGloves, upperGloves)
    ### Open Operation
    kernel = np.ones((5, 5),np.uint8)
    glovesMask = cv2.morphologyEx(glovesMask, cv2.MORPH_OPEN, kernel)
    ### Bitwise PCB&Gloves Mask
    intersectionMask = cv2.bitwise_and(PCBMask, glovesMask)

    if np.sum(intersectionMask) > 0:
        touchFlag = True
    else:
        touchFlag = False
    # cv2.imshow('PCBMask', PCBMask)
    # cv2.imwrite('PCBMask.jpg', PCBMask)
    # cv2.waitKey(0)   
    # cv2.imshow('glovesMask', glovesMask)
    # cv2.imwrite('glovesMask.jpg', glovesMask)
    # cv2.waitKey(0)    
    # cv2.imshow('intersectionMask', intersectionMask)
    # cv2.imwrite('intersectionMask.jpg', intersectionMask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return touchFlag, intersectionMask

def save_result(img, savePath, vid_cap, vid_writer, mode, view_img, save_img):
    """save image or video results

    Args:
        img ([type]): result image
        savePath ([type]): save path
        vid_cap ([type]): video capture
        vid_writer ([type]): video writer
        mode ([type]): image or video mode
        view_img ([type]): displaying flag
        save_img ([type]): saving flag
    """    
    # Stream results
    if view_img:
        cv2.imshow("display", img)
        # cv2.waitKey(1)  # 1 millisecond
    videoName = 'record.avi'
    # Save results (image with detections)
    if save_img:
        if mode == 'image':
            cv2.imwrite(savePath, img)
        else:  # 'video'
            if vid_writer == None:  # new video
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                # fps = min(15.0, vid_cap.get(cv2.CAP_PROP_FPS))   # NX limit 15 FPS
                # fps = vid_cap.get(cv2.CAP_PROP_FPS)   # NX limit 15 FPS
                # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # vid_writer = cv2.VideoWriter(savePath, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer = cv2.VideoWriter(savePath, cv2.VideoWriter_fourcc(*fourcc), 12, (1280, 720))
                # vid_writer = cv2.VideoWriter(os.path.join(savePath, videoName), cv2.VideoWriter_fourcc(*fourcc), 12, (1280, 720))
            vid_writer.write(img)
    return vid_writer

def parse_all_argument():
    """Define All Parse Argument
    """    
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='weights/hands_yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/image_output', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='data/one_image', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='rtsp://192.168.137.97/h265', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--view-img', default=True, help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    
    return  parser.parse_args()


if __name__ == '__main__':

    opt = parse_all_argument()
    print(opt)

    ### Detect Start
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()