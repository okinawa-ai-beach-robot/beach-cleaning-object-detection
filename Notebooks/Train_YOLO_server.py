#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys, os
from pprint import pprint
import shutil
from subprocess import call

# Define dataset:
# Unzipped download from roboflow webpage:
#dataset_location = "../Datasets/beach-cleaning-object-detection.v1i.yolov5pytorch/"
#dataset_location = "../Datasets/beach-cleaning-object-detection.v2i.yolov5pytorch/"
# or; dataset_location = "../Datasets/beach-cleaning-object-detection.v1i.yolov5pytorch/"

# v2 + data augmentation: approx 3x500 samples, 6 classes:
#dataset_location = "../Datasets/beach-cleaning-object-detection.v3-augmented_ver.2.yolov5pytorch/"

# Extended dataset with robot images and further images recorded, state September 9th 2024
dataset_location = "../Datasets/beach-cleaning-object-detection.v8-yolotrain.yolov5pytorch/"

# Define model variant:
yolo_modeltype = "yolov5s"

# Define path to yolo github repository:
yolo_path = "../yolov5/"

# Generate dataset id string for result folder
for did in reversed(dataset_location.split("/")):
    if len(did)>0:
        dataset_id=did.replace(".","__")
        break



# Add yolo code folder to python, and import their utils class:
sys.path.append(yolo_path) 
try:
    import utils
except Exception as ex:
    print("Error:", ex, "\n\n")
    print("Check if current path is in folder 'Networks', yolov5 was downloaded from git, and that yolov5 is located in correct (git-root) folder.")
    print("It may be necessary to click Kernel->Restart and try again...")
    


# ## Read dataset information:

# In[ ]:


# Read number of classes based on YAML config file:
import yaml
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
    


# ## Update path information in dataset for yolo training:

# In[ ]:


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


# In[ ]:


# This is the final dataset configuration file:
file = open(dataset_location + "/data.yaml", "r")
content=file.read()
print("Modified dataset config file is:")
print(content)
file.close()


# In[ ]:


# load the model configuration we will use

with open(yolo_path + "/models/" + yolo_modeltype + ".yaml", 'r') as stream:
    model_def = yaml.safe_load(stream)

print("Original model configuration is:")
pprint(model_def)


# ## Write model configuration file for yolo training:



# modify number of classes for training:
model_def['nc']=num_classes
    
# write modified config:
with open(yolo_path + "/models/beachbot_" + yolo_modeltype + ".yaml", 'w') as file:
    yaml.dump(model_def, file)
    
print("Modified model configuration is:")
pprint(model_def)


# ## Start training and collect models
# * trained models are stored in folder: beach-cleaning-object-detection/Models/




eval_img_widths = [1280, 640, 320, 160]
num_epochs = 800

for img_width in reversed(eval_img_widths):
    targetfolder = "beachbot_" + yolo_modeltype + "_" + dataset_id + "_" + str(img_width)
    targetpath = yolo_path + "runs/train/" + targetfolder + "_results/"
    resultspath = "../Models/" + targetfolder + "/"
    #check if solder exists -> skip computation!
    if os.path.isdir(targetpath) or os.path.isdir(resultspath):
        print("Skipping run", img_width, ":", targetfolder, " or", resultspath,  " (resultfolders) exist!")
    else:
        print("Start run", img_width)
        params=("--img " + str(img_width) + " --batch 16 --epochs " + str(num_epochs) + " --data " + dataset_location + "/data.yaml --cfg " + yolo_path + "/models/beachbot_" + yolo_modeltype + ".yaml --weights " +  yolo_modeltype + ".pt --name " + targetfolder + "_results  --cache").split()
        call(["python", yolo_path+"train.py"] + params)
        
        # Training done
        # Following files in targetfolder are important:
        # weights/best.pt 
        # weights/last.pt
        # results.csv
        resultfiles = ["weights/best.pt","weights/last.pt","results.csv"]

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
        img_heigt_export = img_heigt_export
    )
    with open(resultspath + "export_info.yaml", 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # - Export model to onnx
    # - Export to tensorRT
    # - Export to tensorflow.js
    #https://github.com/NVIDIA/TensorRT/blob/main/quickstart/SemanticSegmentation/tutorial-runtime.ipynb
    #if not os.path.isfile(resultspath + "best.onnx") or not os.path.isfile(resultspath + "best.engine") or not os.path.isdir(resultspath + "best_web_module/"):
    print("Export run", img_width)
    params_export = ("--weights " + resultspath + "best.pt  --img " + str(img_heigt_export) + " " + str(img_width_export) + " --data " + dataset_location + " --include onnx tfjs engine" + " --opset 12 --simplify --device 0 --half").split() 
    call(["python", yolo_path+"export.py"] + params_export)
        
        
        
            
            
        

