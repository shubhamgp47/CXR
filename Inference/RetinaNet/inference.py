from mmengine.config import Config
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.structures import DetDataSample
import os
import cv2

import torch, torchvision
print("torch version:",torch.__version__, "cuda:",torch.cuda.is_available())
import mmdet

print("mmdetection:",mmdet.__version__)
import mmcv

print("mmcv:",mmcv.__version__)


import mmengine
print("mmengine:",mmengine.__version__)

import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
config_file = '/home/woody/iwi5/iwi5204h/mmdetection/VindrMRetinaNet.py'
checkpoint_file = '/home/woody/iwi5/iwi5204h/work_dirs/VindrMRetinaNet/epoch_37.pth'

register_all_modules()
#Load model
model = init_detector(config_file, checkpoint_file, device='cuda:0')

#images
image_folder = '/home/woody/iwi5/iwi5204h/mmdetection/data/VinDir/Test/Converted_to_jpeg/'

#Labels
text_prompts = [
    'infiltration', 'lung opacity', 'lung cyst', 'rib fracture', 
    'pleural effusion', 'lung cavity', 'calcification', 
    'mediastinal shift', 'atelectasis', 'cardiomegaly', 'emphysema', 
    'eventration', 'pulmonary fibrosis', 'consolidation', 'no finding', 
    'pneumothorax', 'other lesion', 'pleural thickening', 
    'aortic enlargement', 'enlarged pa', 'clavicle fracture', 
    'ild', 'nodule/mass', 'edema'
]

image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

#Highest score per label original
'''def get_highest_scores(result):
    highest_scores = {}
    for label, score in zip(result.pred_instances.labels, result.pred_instances.scores):
        label_name = text_prompts[label]
        if label_name not in highest_scores or score > highest_scores[label_name]:
            highest_scores[label_name] = score
    return highest_scores'''

# To debug what labels are out of range
'''def get_highest_scores(result):
    highest_scores = {}
    for label, score in zip(result.pred_instances.labels, result.pred_instances.scores):
        print(f"Label: {label}, Score: {score}")  # Debugging print statement
        if label >= len(text_prompts):
            print(f"Error: Label {label} is out of range for text_prompts with length {len(text_prompts)}")
            continue  # Skip this label if it's out of range
        label_name = text_prompts[label]
        if label_name not in highest_scores or score > highest_scores[label_name]:
            highest_scores[label_name] = score
    return highest_scores'''

# Skip those labels
def get_highest_scores(result):
    highest_scores = {}
    for label, score in zip(result.pred_instances.labels, result.pred_instances.scores):
        if label >= len(text_prompts):
            print(f"Warning: Label {label} is out of range for text_prompts with length {len(text_prompts)}")
            continue  # Skip this label if it's out of range
        label_name = text_prompts[label]
        if label_name not in highest_scores or score > highest_scores[label_name]:
            highest_scores[label_name] = score
    return highest_scores

#Inference
results = []
for image_path in image_files:
    image = mmcv.imread(image_path, channel_order='rgb')
    result = inference_detector(model, image, text_prompt=text_prompts)
    highest_scores = get_highest_scores(result)
    results.append((image_path, highest_scores))

output_directory = '/home/woody/iwi5/iwi5204h/mmdetection/WorkingFolder/Result_Metrics/RetinaNet/'
output_file = output_directory + 'inference_results.txt'

# Print and save results
with open(output_file, 'w') as f:
    for image_path, highest_scores in results:
        print(f"Results for {image_path}:")
        f.write(f"Results for {image_path}:\n")
        for label, score in highest_scores.items():
            print(f"{label}: {score:.4f}")
            f.write(f"{label}: {score:.4f}\n")
        f.write("\n")

print("Results saved to inference_results.txt")