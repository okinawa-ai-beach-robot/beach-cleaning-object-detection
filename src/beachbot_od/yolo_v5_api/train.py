#!/usr/bin/env python
# coding: utf-8


import os
from pprint import pprint
import shutil
import argparse
import yaml
import subprocess

import yolov5
import yolov5.export
import yolov5.train

from beachbot_od.config import BEACHBOT_DATASETS
from beachbot_od.roboflow_api import get_dataset


class Train:
    def __init__(self, args):
        self.dataset_location = args.dataset_location
        self.data_cfg = None
        self.num_classes = None
        self.list_classes = None
        self.data_cfg = None
        self.model_def = None
        self.out_folder = None
        self.targetfolder = None
        self.targetpath = None
        self.resultspath = None
        self.yolo_modeltype = args.yolo_modeltype
        self.fintune = args.fintune
        self.force_retrain = args.force_retrain
        self.img_widths = args.img_widths
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_img_widths = args.eval_img_widths
        self.dataset_id = args.dataset_id
        self.out_folder = os.path.abspath(args.output_folder)
        self.yolo_path = os.path.dirname(yolov5.__file__)

        get_dataset(model_format="yolov5pytorch", location=self.dataset_location)

        # Generate dataset id string for result folder
        for did in reversed(self.dataset_location.split("/")):
            if len(did) > 0:
                dataset_id = did.replace(".", "__")
                break

        # Read number of classes based on YAML config file:
        with open(self.dataset_location + "/data.yaml", "r") as stream:
            data_cfg = yaml.safe_load(stream)
            num_classes = str(data_cfg["nc"])
            list_classes = data_cfg["names"]

        # Print information:
        print("Current configuration:")
        print("Dataset directory is:", self.dataset_location)
        print("Yolo base directoy is:", self.yolo_path)
        print("Dataset defines", num_classes, "classes ->\n", list_classes)
        print("Dataset id for saved results is:", dataset_id)

        # Update path information in dataset for yolo training:
        #
        # Read current dataset config:
        with open(self.dataset_location + "/data.yaml", "r") as stream:
            data_cfg = yaml.safe_load(stream)

        # Modifiy path relative to our working directory:
        data_cfg["train"] = self.dataset_location + "/train/images"
        data_cfg["val"] = self.dataset_location + "/valid/images"
        data_cfg["test"] = self.dataset_location + "/test/images"

        # Write modified dataset config:
        with open(self.dataset_location + "/data.yaml", "w") as file:
            yaml.dump(data_cfg, file)

        # This is the final dataset configuration file:
        file = open(self.dataset_location + "/data.yaml", "r")
        content = file.read()
        print("Modified dataset config file is:")
        print(content)
        file.close()

        # load the model configuration we will use
        with open(
            self.yolo_path + "/models/" + args.yolo_modeltype + ".yaml", "r"
        ) as stream:
            model_def = yaml.safe_load(stream)

        print("Original model configuration is:")
        pprint(model_def)

        # Write model configuration file for yolo training:
        # modify number of classes for training:
        model_def["nc"] = num_classes

        # write modified config:
        with open(
            self.yolo_path + "/models/beachbot_" + self.yolo_modeltype + ".yaml", "w"
        ) as file:
            yaml.dump(model_def, file)

        print("Modified model configuration is:")
        pprint(model_def)

    def change_img_width(self, img_width: int):
        self.img_width = img_width

    def get_git_revision_hash() -> str:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )

    def train(self):

        targetfolder = (
            "beachbot_"
            + self.yolo_modeltype
            + "_"
            + self.dataset_id
            + "_"
            + str(self.img_width)
        )
        targetpath = "runs/train/" + targetfolder + "_results/"
        resultspath = self.out_folder + "/" + targetfolder + "/"

        # check if solder exists -> skip computation!
        if (
            os.path.isdir(targetpath) or os.path.isdir(resultspath)
        ) and not args.force_retrain:
            print(
                "Skipping run",
                self.img_width,
                ":",
                targetfolder,
                " or",
                resultspath,
                " (resultfolders) exist!",
            )
        else:
            print("Start run", self.img_width)
            yolov5.train.run(
                img=self.img_width,
                batch=16,
                epochs=self.num_epochs,
                data=self.dataset_location + "/data.yaml",
                cfg=self.yolo_path
                + "/models/beachbot_"
                + self.yolo_modeltype
                + ".yaml",
                weights=self.yolo_modeltype + ".pt",
                name=targetfolder + "_results",
                cache=True,
                device=self.device,
            )

            # Training done
            # Following files in targetfolder are important:
            resultfiles = [
                "weights/best.pt",
                "results.csv",
                "opt.yaml",
            ]  # optional: "weights/last.pt"

            # Create results folder
            try:
                os.makedirs(resultspath, exist_ok=True)
                print("Trained model will be saved at:", resultspath)
            except OSError as error:
                print("Trained model will update files in:", resultspath)

            # Save model config in result folder
            with open(resultspath + "model.yaml", "w") as file:
                yaml.dump(self.model_def, file)

            # Copy train result data:
            for fname in resultfiles:
                shutil.copy(targetpath + fname, resultspath)

        # Export best model in ONNX format
        # export image dimension must be multiple of 32:
        img_width_export = int((self.img_width // 32) * 32)
        img_heigt = round((self.img_width * 800.0) / 1280.0)
        img_heigt_export = int((img_heigt // 32) * 32)
        # write file with export information
        data = dict(
            img_width=self.img_width,
            img_width_export=img_width_export,
            img_heigt=img_heigt,
            img_heigt_export=img_heigt_export,
            train_args=str(args),
            train_version=self.get_git_revision_hash(),
        )
        with open(resultspath + "export_info.yaml", "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

        # - Export model to onnx
        # - TODO Export to tensorRT
        # - TODO Export to tensorflow.js
        # https://github.com/NVIDIA/TensorRT/blob/main/quickstart/SemanticSegmentation/tutorial-runtime.ipynb
        # if not os.path.isfile(resultspath + "best.onnx") or not os.path.isfile(resultspath + "best.engine") or not os.path.isdir(resultspath + "best_web_module/"):
        print("Export run", self.img_width)
        yolov5.export.run(
            weights=resultspath + "best.pt",
            img=(img_heigt_export, img_width_export),
            data=self.dataset_location,
            include=[
                "onnx",
            ],
            opset=12,
            simplify=True,
            device=self.device,
            half=args.half_precision,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="a script to do stuff")
    parser.add_argument(
        "--dataset_location",
        nargs="?",
        help="path to dataset root directory",
        type=str,
        default=BEACHBOT_DATASETS + "/13/yolov5pytorch/",
    )
    parser.add_argument(
        "--yolo_modeltype",
        nargs="?",
        help="yolo model type, e.g. yolov5s",
        type=str,
        default="yolov5s",
    )
    parser.add_argument(
        "--fintune",
        nargs="?",
        help="modify only output layer instead of all weights",
        type=bool,
        default=False,
    )
    parser.add_argument("--img_widths", nargs="*", default=[1280, 640, 320, 160])
    parser.add_argument("--num_epochs", nargs="?", type=int, default=800)
    parser.add_argument(
        "--output_folder",
        nargs="?",
        help="Base folder for storage of trained models",
        type=str,
        default="../Models/",
    )
    parser.add_argument("--force_retrain", nargs="?", type=bool, default=False)
    parser.add_argument(
        "--device",
        nargs="?",
        help="Device can be cpu, mps or 0...n for cuda devices",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--half_precision",
        nargs="?",
        help="Export in half precision format (only available on cuda training device, i.e. when --device 0 or similar is given)",
        type=bool,
        default=False,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    try:
        args = parse_args()
        print(args)
        train = Train(args)
        train.train()
    except AttributeError as ex:
        print("Error: ", ex)
        print(
            "If error is related to FreeType, try downgrading pillow, e.g. pip install Pillow==9.5.0"
        )
