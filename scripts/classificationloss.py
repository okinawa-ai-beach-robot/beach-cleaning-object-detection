from .. import logger

import sys, os

from PIL import Image
from podm import metrics
import numpy as np
# Accroding to: https://pyimagesearch.com/2022/05/02/mean-average-precision-map-using-the-coco-evaluator/
# This ia also helpful: https://github.com/yfpeng/object_detection_metrics/tree/master/src/podm

def IOU(boxes1, boxes2):
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou 

def boxes_xywh_to_xyxy(boxes):
    res = np.copy(boxes[:,:])
    res[:,2] += res[:,0]
    res[:,3] += res[:,1]
    return res

def box_xywh_to_xyxy(boxes):
    res = np.copy(boxes)
    res[2] += res[0]
    res[3] += res[1]
    return res


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



class ClassificationLoss:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _calc_single(box_pred, label_pred, conf_pred, box_gt, label_gt, iou_thresh=0.5):
        # Custom performance evaluation
        iou_res = IOU(box_pred, box_gt)
        num_detected = box_pred.shape[0]
        num_gt = box_gt.shape[0]

        unfoud_classes = list(range(num_gt))

        best_fitting_gt = iou_res.argmax(axis=1)
        FP=0
        FN=0
        TP=0
        reg_loss=0.0
        class_err=0.0
        for bnum in range(num_detected):
            gt_num = best_fitting_gt[bnum]
            reg_loss += 1.0 - iou_res[bnum, gt_num]
            if iou_res[bnum, gt_num]>iou_thresh:
                TP+=1
                if label_pred[bnum]!=label_gt[gt_num]:
                    class_err+=1
                if gt_num in unfoud_classes:
                    unfoud_classes.remove(gt_num)
            else:
                FP +=1
        reg_loss /= num_detected
        class_err /= TP
        FN = len(unfoud_classes)

        prec = TP / (TP+FP)
        recall = TP / (TP+FN)

        return class_err, reg_loss, (TP, FP, FN), (prec, recall)
    


    def calcImage(boxes_pred, classes_pred, confidences_pred, boxes_gt, classes_gt, iou_thresh=0.5):
        # Peform performance analysis on one image
        imnr=1
        for class_id, confidence, box in zip(classes_pred, confidences_pred, boxes_pred):
            box_pred = box_xywh_to_xyxy(box)
            label_pred = class_id+1
            conf_pred = confidence
            boxes_pred.append(metrics.BoundingBox.of_bbox(imnr, label_pred,box_pred[0],box_pred[1],box_pred[2],box_pred[3], conf_pred))

        for box, label in zip(boxes_gt, classes_gt):
            box_gt = box_xywh_to_xyxy(box)
            label_gt = label+1
            boxes_gt.append(metrics.BoundingBox.of_bbox(imnr, label_gt,box_gt[0],box_gt[1],box_gt[2],box_gt[3], 1.0))


        results = metrics.get_pascal_voc_metrics(boxes_gt, boxes_pred, iou_thresh)

        classes = list(results.keys())
        for clid in classes:
            results[clid].label=results[clid].label-1

        return results


        
    @staticmethod
    def calcDataset(dataset, model, iou_thresh=0.5, verbose=True, progress=True):
        # Perform performance analysis on dataset
        boxes_pred=[]
        boxes_gt=[]

        num_images = len(dataset.images)

        for imnr in range(num_images):
            if progress:
                printProgressBar(imnr, num_images, prefix = 'Progresss:', suffix = 'Complete', length = 50)

            imgpath = dataset.images[imnr]
            rects = dataset.rects[imnr]
            im = Image.open(imgpath)
            im_class_nr = [r['classid'] for r in rects]
            im_class = [dataset.classes[r['classid']] for r in rects]
            im_roi = [r['rect'] for r in rects]

            frame = np.array(im, dtype=model.dtype)
            class_ids, confidences, boxes = model.apply_model(frame)
            for class_id, confidence, box in zip(class_ids, confidences, boxes):
                box_pred = box_xywh_to_xyxy(box)
                label_pred = class_id+1
                conf_pred = confidence
                boxes_pred.append(metrics.BoundingBox.of_bbox(imnr, label_pred,box_pred[0],box_pred[1],box_pred[2],box_pred[3], conf_pred))

            for box, label in zip(im_roi, im_class_nr):
                box_gt = box_xywh_to_xyxy(box)
                label_gt = label+1
                boxes_gt.append(metrics.BoundingBox.of_bbox(imnr, label_gt,box_gt[0],box_gt[1],box_gt[2],box_gt[3], 1.0))


        results = metrics.get_pascal_voc_metrics(boxes_gt, boxes_pred, iou_thresh)

        # remove class offset, offset added as some algorithms ingore class-0 as background
        classes = list(results.keys())
        for clid in classes:
            results[clid].label=results[clid].label-1

        # Print results if requested
        if verbose:
            for cls, metric in results.items():
                label = metric.label
                print("class", dataset.classes[label], " ("+str(label)+")")
                print('ap', metric.ap)
                print('precision', metric.precision)
                print('interpolated_recall', metric.interpolated_recall)
                print('interpolated_precision', metric.interpolated_precision)
                print('tp', metric.tp)
                print('fp', metric.fp)
                print('num_groundtruth', metric.num_groundtruth)
                print('num_detection', metric.num_detection)
                print("--------------------")

        return results


        


