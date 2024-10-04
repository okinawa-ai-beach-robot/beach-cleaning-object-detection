import os
import beachbot
import numpy as np




# Load dataset from path:
data_path = "/saion/Deploy/beachbot/beach-cleaning-object-detection/Datasets/beach-cleaning-object-detection.v8-yolotrain.yolov5pytorch"
# Alternative: list installed datasets data_path = beachbot.ai.Dataset.list_dataset_paths()[0]
print("Dataset path is", data_path)
dataset = beachbot.ai.Dataset(data_path, "test").random_prune(num_samples=15)


# Load AI detector from path:
model_path = "/saion/Deploy/beachbot/beach-cleaning-object-detection/Models/beachbot_yolov5s_beach-cleaning-object-detection__v8-yolotrain__yolov5pytorch_1280"
# Alternative: list installed models model_file = beachbot.ai.DerbrisDetector.list_model_paths()
print("Model path is", model_path)
model_type = beachbot.ai.DerbrisDetector.get_model_type(model_path)
print("Model type is", model_type)
model_cls_list= beachbot.ai.DerbrisDetector.list_models_by_type(model_type)
print("Model classes are", model_cls_list)
model_cls = model_cls_list[0]
model = model_cls(model_file=model_path, use_accel=False)



r1 = dataset.rects[0][0]['rect']
c1 = dataset.classes[0]


evalresult =beachbot.utils.ClassificationLoss.calcDataset(dataset, model, verbose=False)

for cls, metric in evalresult.items():
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