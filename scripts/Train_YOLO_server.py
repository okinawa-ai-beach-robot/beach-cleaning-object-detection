#!/usr/bin/env python
# coding: utf-8






import sys, os
from pprint import pprint
import shutil
from subprocess import call
import argparse
import sys
import yaml
import shutil
import subprocess

import yolov5
from yolov5 import train, val, detect, export, utils
import yolov5.export
import yolov5.train


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


# Define path to yolo github repository that is equal to the pip base folder:
yolo_path = os.path.dirname(yolov5.__file__)


def parse_args():
    parser=argparse.ArgumentParser(description="a script to do stuff")
    parser.add_argument("--dataset_location", nargs='?', help="path to dataset root directory", type=str, default="../Datasets/trash_detection.minimal_example.yolov5pytorch/")
    parser.add_argument("--yolo_modeltype", nargs='?', help="yolo model type, e.g. yolov5s", type=str, default="yolov5s")
    parser.add_argument("--fintune", nargs='?', help="modify only output layer instead of all weights", type=bool, default=False)
    parser.add_argument('--img_widths', nargs="*", default=[1280, 640, 320, 160])
    parser.add_argument("--num_epochs", nargs='?', type=int, default=800)
    parser.add_argument("--output_folder", nargs='?', help="Base folder for storage of trained models", type=str, default="../Models/")
    parser.add_argument("--force_retrain", nargs='?', type=bool, default=False)
    parser.add_argument("--device", nargs='?', help="Device can be cpu, mps or 0...n for cuda devices", type=str, default="cpu")
    parser.add_argument("--half_precision", nargs='?', help="Export in half precision format (only available on cuda training device, i.e. when --device 0 or similar is given)", type=bool, default=False)

    
    
    args=parser.parse_args()
    return args




def modeltrain():
    args=parse_args()
    print(args)


    # Define dataset folder:
    #
    # Unzipped download from roboflow webpage:
    #dataset_location = "../Datasets/beach-cleaning-object-detection.v1i.yolov5pytosrch/"
    #dataset_location = "../Datasets/beach-cleaning-object-detection.v2i.yolov5pytorch/"
    # or; dataset_location = "../Datasets/beach-cleaning-object-detection.v1i.yolov5pytorch/"
    #
    # v2 + data augmentation: approx 3x500 samples, 6 classes:
    #dataset_location = "../Datasets/beach-cleaning-object-detection.v3-augmented_ver.2.yolov5pytorch/"
    #
    # Extended dataset with robot images and further images recorded, state September 9th 2024
    # e.g. `dataset_location = "../Datasets/beach-cleaning-object-detection.v8-yolotrain.yolov5pytorch/"
    #       dataset_location = "../Datasets/trash_detection.minimal_example.yolov5pytorch/"
    dataset_location = os.path.abspath(args.dataset_location)

    # Define model variant:
    # e.g. yolo_modeltype = "yolov5s"
    yolo_modeltype = args.yolo_modeltype



    # Generate dataset id string for result folder
    for did in reversed(dataset_location.split("/")):
        if len(did)>0:
            dataset_id=did.replace(".","__")
            break




    # Read number of classes based on YAML config file:
    with open(dataset_location + "/data.yaml", 'r') as stream:
        data_cfg = yaml.safe_load(stream)
        num_classes = str(data_cfg['nc'])
        list_classes = data_cfg['names']

    # Print information:
    print("Current configuration:")
    print("Dataset directory is:", dataset_location)
    print("Yolo base directoy is:", yolo_path)    
    print("Dataset defines", num_classes, "classes ->\n", list_classes)
    print ("Dataset id for saved results is:", dataset_id)
        


    # Update path information in dataset for yolo training:
    #
    # Read current dataset config:
    with open(dataset_location + "/data.yaml", 'r') as stream:
        data_cfg = yaml.safe_load(stream)
        
    # Modifiy path relative to our working directory:
    data_cfg['train']=dataset_location + "/train/images"
    data_cfg['val']=dataset_location + "/valid/images"
    data_cfg['test']=dataset_location + "/test/images"
        
    # Write modified dataset config:
    with open(dataset_location + "/data.yaml", 'w') as file:
        yaml.dump(data_cfg, file)




    # This is the final dataset configuration file:
    file = open(dataset_location + "/data.yaml", "r")
    content=file.read()
    print("Modified dataset config file is:")
    print(content)
    file.close()



    # load the model configuration we will use
    with open(yolo_path + "/models/" + yolo_modeltype + ".yaml", 'r') as stream:
        model_def = yaml.safe_load(stream)

    print("Original model configuration is:")
    pprint(model_def)


    # Write model configuration file for yolo training:
    # modify number of classes for training:
    model_def['nc']=num_classes
        
    # write modified config:
    with open(yolo_path + "/models/beachbot_" + yolo_modeltype + ".yaml", 'w') as file:
        yaml.dump(model_def, file)
        
    print("Modified model configuration is:")
    pprint(model_def)


    # Start training and collect models
    eval_img_widths = args.img_widths
    num_epochs = args.num_epochs
    device=args.device

    out_folder = os.path.abspath(args.output_folder)

    for img_width in reversed(eval_img_widths):
        img_width = int(img_width)
        targetfolder = "beachbot_" + yolo_modeltype + "_" + dataset_id + "_" + str(img_width)
        targetpath = "runs/train/" + targetfolder + "_results/"
        resultspath = out_folder + "/" + targetfolder + "/"
        #check if solder exists -> skip computation!
        
        if (os.path.isdir(targetpath) or os.path.isdir(resultspath)) and not args.force_retrain:
            print("Skipping run", img_width, ":", targetfolder, " or", resultspath,  " (resultfolders) exist!")
        else:
            # if yolo output folder exists, delete it, as yolo will add suffixes -1..-n to the results folder otherwise:
            if os.path.isdir(targetpath): 
                shutil.rmtree(targetpath)
            print("Start run", img_width)
            yolov5.train.run(img=img_width, batch=16, epochs=num_epochs, data=dataset_location + "/data.yaml", cfg=yolo_path + "/models/beachbot_" + yolo_modeltype + ".yaml", weights=yolo_modeltype + ".pt", name=targetfolder + "_results", cache=True, device=device  )

            # Training done
            # Following files in targetfolder are important:
            resultfiles = ["weights/best.pt","results.csv", "opt.yaml"] # optional: "weights/last.pt"

            # Create results folder
            try: 
                os.makedirs(resultspath, exist_ok = True) 
                print("Trained model will be saved at:", resultspath) 
            except OSError as error: 
                print("Trained model will update files in:", resultspath) 

            # Save model config in result folder
            with open(resultspath + "model.yaml", 'w') as file:
                yaml.dump(model_def, file)

            # Copy train result data:
            for fname in resultfiles:
                shutil.copy(targetpath + fname, resultspath)

        # Export best model in ONNX format
        # export image dimension must be multiple of 32:
        img_width_export = int((img_width//32)*32)
        img_heigt =  round((img_width*800.0)/1280.0)
        img_heigt_export = int((img_heigt//32)*32)
        #write file with export information
        data = dict(
            img_width = img_width,
            img_width_export = img_width_export,
            img_heigt = img_heigt,
            img_heigt_export = img_heigt_export,
            train_args = str(args),
            train_version = get_git_revision_hash()
        )
        with open(resultspath + "export_info.yaml", 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

        # - Export model to onnx
        # - TODO Export to tensorRT
        # - TODO Export to tensorflow.js
        #https://github.com/NVIDIA/TensorRT/blob/main/quickstart/SemanticSegmentation/tutorial-runtime.ipynb
        #if not os.path.isfile(resultspath + "best.onnx") or not os.path.isfile(resultspath + "best.engine") or not os.path.isdir(resultspath + "best_web_module/"):
        print("Export run", img_width)
        yolov5.export.run(weights=resultspath + "best.pt", img=(img_heigt_export, img_width_export), data=dataset_location, include=["onnx",], opset=12, simplify=True, device=device, half=args.half_precision )
            
            
                
                
            





if __name__ == '__main__':
    try:
        modeltrain()
    except AttributeError as ex:
        print("Error: ", ex)
        print("If error is related to FreeType, try downgrading pillow, e.g. pip install Pillow==9.5.0")





