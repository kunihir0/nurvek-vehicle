!pip install -q ultralytics
The history saving thread hit an unexpected error (OperationalError('attempt to write a readonly database')).History will not be written to the database.
DEPENDENCIES
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from xml.dom import minidom
import random
import easyocr
from ultralytics import YOLO
import yaml
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import pytesseract
from PIL import Image
Creating new Ultralytics Settings v0.0.6 file ✅ 
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
convert_xml2yolo Function
image.png

Overview
The convert_xml2yolo function is designed to convert object detection annotations from XML format (commonly used in datasets like Pascal VOC) to YOLO format. This conversion is essential for training object detection models using the YOLO framework, which requires annotations in a specific format.

Function Signature
def convert_xml2yolo(lut, input_path, output_path):
Parameters
lut: A dictionary that maps class names (strings) to numeric labels (integers). This is used to assign YOLO-compatible class IDs to the objects in the annotations.
input_path: The directory containing the XML annotation files.
output_path: The directory where the converted YOLO annotation files will be saved.
Workflow
Check Output Directory:

If the specified output_path does not exist, it is created using os.mkdir.
Iterate Over XML Files:

The function uses the glob module to find all XML files in the input_path directory.
Parse XML File:

Each XML file is parsed using the minidom module to extract annotation details.
Extract Image Dimensions:

The width and height of the image are retrieved from the <size> tag in the XML file.
Process Each Object:

For each <object> tag in the XML file:
The class name is mapped to a numeric label using the lut dictionary.
The bounding box coordinates (xmin, ymin, xmax, ymax) are extracted and converted to YOLO format using the convert_coordinates helper function.
Write YOLO Annotation File:

The converted annotations are written to a .txt file in the output_path directory.
Completion Message:

A message is printed to indicate that the conversion process is finished.
Helper Function: convert_coordinates
The convert_coordinates function is used to transform bounding box coordinates from absolute pixel values to normalized YOLO format.

Formula
Given:

Image dimensions: (width, height)
Bounding box: (xmin, xmax, ymin, ymax)
The YOLO format is calculated as:

x_center = (xmin + xmax) / 2 / width
y_center = (ymin + ymax) / 2 / height
box_width = (xmax - xmin) / width
box_height = (ymax - ymin) / height
Code
def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
Example Usage
lut = {
    "car": 0,
    "person": 1,
    "bicycle": 2
}

convert_xml2yolo(
    lut,
    input_path="/path/to/xml/annotations",
    output_path="/path/to/yolo/annotations"
)
Notes
If a class name in the XML file is not found in the lut dictionary, it is assigned a default label of 0.
The function assumes that the XML files follow the Pascal VOC format.
Output Format
Each line in the YOLO annotation file corresponds to an object and follows the format:

<class_id> <x_center> <y_center> <width> <height>
All values are normalized to the range [0, 1].

Conclusion
The convert_xml2yolo function simplifies the process of preparing annotation data for YOLO-based object detection models. By automating the conversion from XML to YOLO format, it ensures compatibility and reduces manual effort.

# -*- coding: utf-8 -*-

lut={}
lut["accessory"] =0
lut["top"]       =1
lut["bottom"]    =2
lut["bag"]       =3
lut["shoes"]     =4



def convert_coordinates(size, box):
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1]-box[0]
    h = box[3]-box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def convert_xml2yolo( lut,input_path, output_path ):

    if not os.path.exists(output_path):
         os.mkdir(output_path) 
        
    for fname in glob(f"{input_path}/*.xml"):
        
        xmldoc = minidom.parse(fname)
        annot_fname = fname.split("/")[-1][:-4]
        fname_out = f"{output_path}/{annot_fname}.txt"
        
        with open(fname_out, "w") as f:
            
            itemlist = xmldoc.getElementsByTagName('object')
            size = xmldoc.getElementsByTagName('size')[0]
            width = int((size.getElementsByTagName('width')[0]).firstChild.data)
            height = int((size.getElementsByTagName('height')[0]).firstChild.data)

            for item in itemlist:
                # get class label
                classid =  (item.getElementsByTagName('name')[0]).firstChild.data
                if classid in lut:
                    label_str = str(lut[classid])
                else:
                    label_str = "0"
                    #print ("warning: label '%s' not in look-up table" % classid)

                # get bbox coordinates
                xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = convert_coordinates((width,height), b)
                #print(bb)

                f.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')

        #print ("wrote %s" % fname_out)
    print("Converting is finished!")

convert_xml2yolo( lut, input_path = "/kaggle/input/car-plate-detection/annotations", output_path = "/kaggle/working/annotations")
Converting is finished!
VAL & TRAIN SPLIT PATHS
def create_directories_if_not_exist(directories: list):
    # Assert: Is the parameter a list?
    assert isinstance(directories, list), "The parameter must be a list (array)."
    
    # Assert: Is every element in the list a string?
    for dir_path in directories:
        assert isinstance(dir_path, str), f"The element '{dir_path}' in the list is not a string."

    # Create directories
    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
            print(f"Created: {dir_path}")
        else:
            print(f"Already exists: {dir_path}")
directories = [
    "/kaggle/working/car_plate_dataset",
    "/kaggle/working/car_plate_dataset/images",
    "/kaggle/working/car_plate_dataset/labels",
    "/kaggle/working/car_plate_dataset/images/train",
    "/kaggle/working/car_plate_dataset/images/val",
    "/kaggle/working/car_plate_dataset/labels/train",
    "/kaggle/working/car_plate_dataset/labels/val"
]

create_directories_if_not_exist(directories)
Created: /kaggle/working/car_plate_dataset
Created: /kaggle/working/car_plate_dataset/images
Created: /kaggle/working/car_plate_dataset/labels
Created: /kaggle/working/car_plate_dataset/images/train
Created: /kaggle/working/car_plate_dataset/images/val
Created: /kaggle/working/car_plate_dataset/labels/train
Created: /kaggle/working/car_plate_dataset/labels/val
split_images_and_labels Function
Overview
The split_images_and_labels function splits a dataset of images and labels into training and validation sets, useful for machine learning tasks like object detection.

Parameters
val_size (float): Proportion of the dataset for validation (default: 0.1).
input_dir (str): Directory containing images/ and annotations/ subfolders.
output_dir (str): Directory where the split dataset will be saved.
Workflow
Splitting: Images are divided into train and val sets based on val_size.
Copying: Files are copied to output_dir under images/train, images/val, labels/train, and labels/val.
Completion: Prints a message when done.
Example Usage
split_images_and_labels(
    val_size=0.2,
    input_dir="/path/to/input",
    output_dir="/path/to/output"
)
Notes
Assumes matching filenames for images and labels (e.g., image1.jpg and image1.txt).
Creates output_dir if it doesn’t exist.
Output Example
For val_size=0.5:

plaintext
output_dir/
    images/
        train/
            car1.jpg
        val/
            car2.jpg
    labels/
        train/
            car1.txt
        val/
            car2.txt
def split_images_and_labels(val_size = 0.1, input_dir = "/kaggle/input/car-plate-detection",
                           output_dir = "/kaggle/working/car_plate_dataset"):
    
        img_path = os.listdir(f"{input_dir}/images")
        label_path = os.listdir(f"{input_dir}/annotations")
        imgs_length = len(img_path)
        for i,img in enumerate(img_path):
            spname = "train" if i < int(imgs_length*float(1 - val_size)) else "val"
            
            #print(spname)
            
            os.system(f"cp {input_dir}/images/{img} {output_dir}/images/{spname}/{img}")
            #print(f"the image {input_dir}/images/{img} image copied to {output_dir}/images/{spname}/{img}")
            
            os.system(f"cp /kaggle/working/annotations/{img.split('.')[0]}.txt {output_dir}/labels/{spname}/{img.split('.')[0]}.txt")
            #print(f"the label /kaggle/working/annotations/{img.split('.')[0]}.txt image copied to {output_dir}/labels/{spname}/{img.split('.')[0]}.txt")  
    
        print("Splitting is finished!")
    
split_images_and_labels()
Splitting is finished!
YAML PARAMETERS
# Data to be written to the YAML file
data = {
    'train': '/kaggle/working/car_plate_dataset/images/train',
    'val': '/kaggle/working/car_plate_dataset/images/val',
    'names': {
        0: "number_plate"
    } # List formatında olmalı!
}

# Writing the data to a YAML file
with open('data.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)

print("Data has been written to 'data.yaml'")
Data has been written to 'data.yaml'
Visualization
Overview
This script visualizes a grid of images with bounding boxes drawn from YOLO annotations. It is useful for verifying the correctness of annotations and inspecting the dataset.

Workflow
File Collection:

Collects image files from the specified directory (image_dir) with supported extensions (*.jpg, *.jpeg, *.png).
Selects the first 9 images for visualization.
Grid Creation:

Creates a 3x3 grid using Matplotlib to display the images.
Bounding Box Drawing:

For each image, the corresponding YOLO annotation file is read.
Bounding boxes are converted from YOLO format to pixel coordinates and drawn on the image.
Display:

Displays the grid of images with bounding boxes using Matplotlib.
Key Variables
image_dir: Path to the directory containing the images.
label_dir: Path to the directory containing the YOLO annotation files.
image_extensions: List of supported image file extensions.
Example Usage
# File paths
image_dir = "/path/to/images"
label_dir = "/path/to/labels"

# Supported extensions
image_extensions = ["*.jpg", "*.jpeg", "*.png"]

# Visualization script
# (Refer to the script in `main.py` for full implementation)
Notes
The script assumes that the filenames of the images and their corresponding labels match (e.g., image1.jpg and image1.txt).
Bounding boxes are drawn in red for better visibility.
Ensure that the image_dir and label_dir paths are correctly set before running the script.
Output Example
The script generates a 3x3 grid of images with bounding boxes drawn, as shown below:

+---------+---------+---------+
| Image 1 | Image 2 | Image 3 |
+---------+---------+---------+
| Image 4 | Image 5 | Image 6 |
+---------+---------+---------+
| Image 7 | Image 8 | Image 9 |
+---------+---------+---------+
Bounding boxes are overlaid on the images in red.

# File paths
image_dir = "/kaggle/working/car_plate_dataset/images/train"
label_dir = "/kaggle/working/car_plate_dataset/labels/train"

# Supported extensions
image_extensions = ["*.jpg", "*.jpeg", "*.png"]

# Collect all image files
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob(os.path.join(image_dir, ext)))

# Use the first 9 images
image_paths = sorted(image_paths)[:9]

# Draw a 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for idx, image_path in enumerate(image_paths):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    # Find the corresponding label file (change extension to .txt)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(label_dir, base_name + ".txt")
    
    # Draw with Matplotlib
    ax = axes[idx]
    ax.imshow(img)
    ax.axis("off")
    
    # Draw boxes if labels exist
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
            
            # Convert from YOLO format to pixels
            x1 = int((x_center - box_width / 2) * w)
            y1 = int((y_center - box_height / 2) * h)
            x2 = int((x_center + box_width / 2) * w)
            y2 = int((y_center + box_height / 2) * h)
            
            # Draw rectangle
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

plt.tight_layout()
plt.show()

YOLO Training
Overview of YOLO
YOLO (You Only Look Once) is a state-of-the-art object detection algorithm that performs detection in a single pass through the network. Unlike traditional methods that use region proposals and multiple stages, YOLO treats object detection as a regression problem, predicting bounding boxes and class probabilities directly from the input image.

Key Features of YOLO
Speed: YOLO is extremely fast because it processes the entire image in a single forward pass.
Accuracy: It achieves high accuracy by learning global image context and spatial relationships.
Unified Architecture: YOLO uses a single convolutional neural network (CNN) to predict bounding boxes and class probabilities simultaneously.
Explanation of IoU and mAP Metrics
Intersection over Union (IoU)
IoU measures the overlap between two bounding boxes, typically the predicted and ground truth boxes. It is defined as:
I
o
U
=
A
r
e
a
 
o
f
 
O
v
e
r
l
a
p
A
r
e
a
 
o
f
 
U
n
i
o
n

Area of Overlap: The area where the predicted and ground truth boxes intersect.
Area of Union: The total area covered by both boxes.
IoU ranges from 0 to 1, where 1 indicates perfect overlap. It is used to determine whether a predicted bounding box is a true positive or a false positive.

Mean Average Precision (mAP)
mAP evaluates the performance of object detection models by calculating the average precision (AP) for each class and then taking the mean.

Average Precision (AP)
AP is the area under the Precision-Recall curve for a specific class. It is calculated as:
A
P
=
∫
1
0
P
(
R
)
d
R

P(R): Precision as a function of Recall.
Mean Average Precision
The mAP is then calculated as:
m
A
P
=
1
N
N
∑
i
=
1
 
A
P
i

N: Total number of classes.
AP_i: Average Precision for class 
i
.
Explanation of the Training Code
GPU Configuration
gpu_count = torch.cuda.device_count()
device = list(range(gpu_count)) if gpu_count > 1 else 0
torch.cuda.device_count(): Checks the number of available GPUs.
device: Sets the device to use multiple GPUs if available, otherwise defaults to a single GPU.
Model Initialization
model = YOLO("yolo11n.pt")
YOLO: Initializes the YOLO model with the specified weights file (yolo11n.pt).
Training Configuration
results = model.train(
    data="/kaggle/working/data.yaml",   # Dataset configuration
    epochs=200,                         # 200 epochs
    imgsz=640,                          # Suitable for smaller images
    batch=32,                           # Adjustable based on GPU RAM
    workers=2,                          # Ideal starting value for Tesla T4
    device=device,                      # GPU setting
    augment=True,                       # Default YOLO augmentations are automatically enabled
    patience=20,                        # Stops early if no improvement for 20 epochs
    val=True,                           # Validation is performed at the end of each epoch
)
Key Parameters
data: Path to the dataset configuration file (data.yaml).
epochs: Number of training epochs (200 in this case).
imgsz: Image size for training (640x640 pixels).
batch: Batch size, adjustable based on GPU memory.
workers: Number of data loader workers (2 is ideal for Tesla T4 GPUs).
device: Specifies the GPU(s) to use.
augment: Enables default YOLO augmentations for data augmentation.
patience: Early stopping if no improvement for 20 epochs.
val: Enables validation at the end of each epoch.
Notes
Ensure that the dataset is correctly formatted and the data.yaml file is properly configured.
Adjust the batch size and workers based on the available GPU resources.
Use a pre-trained weights file (e.g., yolo11n.pt) to speed up training and improve accuracy.
Output
The training process outputs:

Model Weights: Saved at regular intervals and at the end of training.
Metrics: Training and validation loss, mAP, and other performance metrics.
Logs: Detailed logs for each epoch, including loss and mAP values.
This documentation provides a comprehensive explanation of the YOLO training process, including the metrics and code used. Let me know if you need further clarifications or additional details!

# Check the number of GPUs
gpu_count = torch.cuda.device_count()
device = list(range(gpu_count)) if gpu_count > 1 else 0

model = YOLO("yolo11n.pt")

results = model.train(
    data="/kaggle/working/data.yaml",   # Dataset configuration
    epochs=200,                         # 200 epochs
    imgsz=640,                          # Suitable for smaller images
    batch=32,                            # Adjustable based on GPU RAM
    workers=2,                          # Ideal starting value for Tesla T4
    device=device,                      # GPU setting

    # ✅ Default Regularization & Optimization (YOLO uses its own defaults)
    # If you don't define these parameters, defaults will be used.

    # ✅ Default Augmentations
    augment=True,                       # Default YOLO augmentations are automatically enabled

    # ✅ Early Stopping and Validation
    patience=20,                        # Stops early if no improvement for 10 epochs
    val=True,                           # Validation is performed at the end of each epoch
)
Ultralytics 8.3.134 🚀 Python-3.11.11 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)
engine/trainer: agnostic_nms=False, amp=True, augment=True, auto_augment=randaugment, batch=32, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/kaggle/working/data.yaml, degrees=0.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=200, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo11n.pt, momentum=0.937, mosaic=1.0, multi_scale=False, name=train8, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=20, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, rect=False, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=runs/detect/train8, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=2, workspace=None
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      
  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1     32096  ultralytics.nn.modules.block.C3k2            [256, 64, 1, False]           
 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1     86720  ultralytics.nn.modules.block.C3k2            [192, 128, 1, False]          
 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]           
 23        [16, 19, 22]  1    430867  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]           
YOLO11n summary: 181 layers, 2,590,035 parameters, 2,590,019 gradients, 6.4 GFLOPs

Transferred 448/499 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3592.1±1080.3 MB/s, size: 499.9 KB)
train: Scanning /kaggle/working/car_plate_dataset/labels/train.cache... 389 images, 0 backgrounds, 0 corrupt: 100%|██████████| 389/389 [00:00<?, ?it/s]
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, num_output_channels=3, method='weighted_average'), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 593.2±110.6 MB/s, size: 400.5 KB)
val: Scanning /kaggle/working/car_plate_dataset/labels/val.cache... 44 images, 0 backgrounds, 0 corrupt: 100%|██████████| 44/44 [00:00<?, ?it/s]
Plotting labels to runs/detect/train8/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.002, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 2 dataloader workers
Logging results to runs/detect/train8
Starting training for 200 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/200      10.7G       1.56      3.795       1.38         12        640: 100%|██████████| 13/13 [00:05<00:00,  2.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.38it/s]
                   all         44         45    0.00326      0.956      0.475      0.261
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/200      10.7G      1.421      2.363      1.235         11        640: 100%|██████████| 13/13 [00:04<00:00,  3.19it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.21it/s]
                   all         44         45    0.00303      0.889      0.253      0.131
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/200      10.7G      1.477      1.977      1.239         13        640: 100%|██████████| 13/13 [00:04<00:00,  3.17it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.51it/s]
                   all         44         45      0.882      0.167      0.409      0.203
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/200      10.7G      1.543      1.885      1.297         11        640: 100%|██████████| 13/13 [00:03<00:00,  3.30it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.33it/s]
                   all         44         45      0.657      0.133      0.306      0.198
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/200      10.7G      1.489      1.735      1.237         10        640: 100%|██████████| 13/13 [00:04<00:00,  3.19it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.34it/s]
                   all         44         45      0.742      0.321       0.55      0.302
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/200      10.7G      1.444      1.593      1.232          7        640: 100%|██████████| 13/13 [00:04<00:00,  3.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.27it/s]
                   all         44         45      0.785        0.6      0.655      0.383
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/200      10.7G      1.423      1.481      1.214         10        640: 100%|██████████| 13/13 [00:04<00:00,  3.11it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.78it/s]
                   all         44         45      0.861      0.276      0.368      0.229
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/200      10.7G      1.451      1.428      1.222         11        640: 100%|██████████| 13/13 [00:03<00:00,  3.33it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.92it/s]
                   all         44         45     0.0364     0.0667     0.0336     0.0111
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/200      10.7G      1.477      1.343      1.236          8        640: 100%|██████████| 13/13 [00:03<00:00,  3.35it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.50it/s]
                   all         44         45    0.00102      0.289    0.00254   0.000996
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/200      10.7G      1.471       1.32      1.239          9        640: 100%|██████████| 13/13 [00:03<00:00,  3.43it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.80it/s]
                   all         44         45      0.591      0.156       0.19     0.0763
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/200      10.7G      1.528      1.306      1.297         12        640: 100%|██████████| 13/13 [00:04<00:00,  3.14it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  1.75it/s]
                   all         44         45      0.365      0.511      0.376      0.195
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/200      10.7G      1.505      1.261      1.264          9        640: 100%|██████████| 13/13 [00:03<00:00,  3.45it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.11it/s]
                   all         44         45      0.386     0.0222     0.0141    0.00456
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/200      10.7G      1.444      1.187       1.25         10        640: 100%|██████████| 13/13 [00:03<00:00,  3.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.15it/s]
                   all         44         45      0.699      0.511      0.563      0.293
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/200      10.7G       1.46      1.189       1.25         10        640: 100%|██████████| 13/13 [00:03<00:00,  3.28it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.65it/s]
                   all         44         45      0.486      0.467      0.382      0.196
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/200      10.7G      1.392      1.097      1.211         10        640: 100%|██████████| 13/13 [00:03<00:00,  3.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.88it/s]
                   all         44         45      0.692      0.711      0.625      0.348
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/200      10.7G       1.44      1.063      1.239         13        640: 100%|██████████| 13/13 [00:04<00:00,  3.17it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.30it/s]
                   all         44         45      0.458      0.333       0.33      0.141
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/200      10.7G      1.441      1.072      1.238         14        640: 100%|██████████| 13/13 [00:04<00:00,  3.19it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.48it/s]
                   all         44         45      0.575      0.511      0.448      0.227
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/200      10.7G       1.39      1.031      1.225          5        640: 100%|██████████| 13/13 [00:04<00:00,  3.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.23it/s]
                   all         44         45      0.689      0.641      0.653      0.402
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/200      10.7G      1.359     0.9603      1.183         10        640: 100%|██████████| 13/13 [00:03<00:00,  3.42it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.41it/s]
                   all         44         45      0.665      0.711       0.73      0.431
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/200      10.7G      1.346     0.9603      1.188          9        640: 100%|██████████| 13/13 [00:03<00:00,  3.44it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.12it/s]
                   all         44         45      0.638      0.622      0.685      0.408
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/200      10.7G      1.373     0.9989      1.208         11        640: 100%|██████████| 13/13 [00:03<00:00,  3.40it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.78it/s]
                   all         44         45      0.789      0.733      0.774      0.465
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/200      10.7G      1.393     0.9847      1.203         14        640: 100%|██████████| 13/13 [00:03<00:00,  3.31it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.02it/s]
                   all         44         45       0.71      0.644      0.652      0.389
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/200      10.7G      1.349     0.9313      1.173         14        640: 100%|██████████| 13/13 [00:03<00:00,  3.46it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.24it/s]
                   all         44         45      0.672        0.6      0.693      0.317
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/200      10.7G       1.37     0.9274      1.183         12        640: 100%|██████████| 13/13 [00:04<00:00,  3.06it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.97it/s]
                   all         44         45       0.83      0.868      0.863      0.514
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/200      10.7G      1.359     0.9131      1.174         13        640: 100%|██████████| 13/13 [00:03<00:00,  3.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.54it/s]
                   all         44         45      0.763      0.867      0.778      0.446
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/200      10.7G      1.347     0.8804      1.191         14        640: 100%|██████████| 13/13 [00:04<00:00,  3.10it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.64it/s]
                   all         44         45      0.809      0.778      0.797      0.485
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/200      10.7G      1.352     0.8714      1.187         13        640: 100%|██████████| 13/13 [00:03<00:00,  3.44it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.45it/s]
                   all         44         45      0.839      0.711      0.758      0.473
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/200      10.7G      1.321     0.9402      1.189          8        640: 100%|██████████| 13/13 [00:03<00:00,  3.49it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.04it/s]
                   all         44         45      0.799      0.844      0.856      0.507
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/200      10.7G      1.313     0.8269      1.161          8        640: 100%|██████████| 13/13 [00:03<00:00,  3.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.18it/s]
                   all         44         45      0.868        0.8      0.812       0.46
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/200      10.7G      1.261     0.8261      1.152         11        640: 100%|██████████| 13/13 [00:04<00:00,  3.03it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.83it/s]
                   all         44         45      0.763      0.844      0.788       0.53
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/200      10.7G      1.269     0.8466      1.171          8        640: 100%|██████████| 13/13 [00:03<00:00,  3.40it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.38it/s]
                   all         44         45      0.796       0.78      0.775      0.506
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/200      10.7G      1.289     0.8137       1.16          8        640: 100%|██████████| 13/13 [00:03<00:00,  3.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.28it/s]
                   all         44         45      0.808      0.689      0.793      0.468
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/200      10.7G      1.259      0.793      1.169          7        640: 100%|██████████| 13/13 [00:04<00:00,  3.22it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.42it/s]
                   all         44         45      0.846      0.756      0.808      0.471
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/200      10.7G      1.287     0.8118      1.144         15        640: 100%|██████████| 13/13 [00:04<00:00,  3.14it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.86it/s]
                   all         44         45      0.937      0.889      0.894      0.528
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/200      10.7G      1.311     0.8498      1.149          4        640: 100%|██████████| 13/13 [00:03<00:00,  3.41it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.33it/s]
                   all         44         45      0.823        0.8      0.819      0.469
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/200      10.7G      1.252     0.8217      1.158          4        640: 100%|██████████| 13/13 [00:04<00:00,  3.06it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.63it/s]
                   all         44         45      0.825      0.735      0.795      0.458
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/200      10.7G      1.279     0.8104      1.148         12        640: 100%|██████████| 13/13 [00:03<00:00,  3.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.95it/s]
                   all         44         45      0.939      0.844       0.87      0.537
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/200      10.7G       1.28     0.8061      1.142         12        640: 100%|██████████| 13/13 [00:04<00:00,  3.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.74it/s]
                   all         44         45      0.884        0.8      0.839      0.525
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/200      10.7G      1.247     0.7914      1.159         10        640: 100%|██████████| 13/13 [00:03<00:00,  3.27it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.84it/s]
                   all         44         45      0.814      0.778      0.772      0.509
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/200      10.7G      1.249     0.7566      1.125          7        640: 100%|██████████| 13/13 [00:03<00:00,  3.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.11it/s]
                   all         44         45      0.872      0.822      0.861      0.532
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/200      10.7G      1.274     0.7917      1.157         14        640: 100%|██████████| 13/13 [00:03<00:00,  3.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.11it/s]
                   all         44         45      0.892      0.778      0.798       0.51
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/200      10.7G      1.288     0.7905      1.174          9        640: 100%|██████████| 13/13 [00:04<00:00,  3.07it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.16it/s]
                   all         44         45      0.838      0.806       0.83       0.49
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/200      10.7G      1.279     0.7781      1.152          6        640: 100%|██████████| 13/13 [00:03<00:00,  3.28it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.03it/s]
                   all         44         45      0.903      0.756      0.825       0.49
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/200      10.7G      1.259     0.7594      1.125         10        640: 100%|██████████| 13/13 [00:04<00:00,  3.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.19it/s]
                   all         44         45      0.785      0.844      0.834      0.528
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/200      10.7G      1.208     0.7433       1.11          9        640: 100%|██████████| 13/13 [00:03<00:00,  3.39it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.27it/s]
                   all         44         45      0.924      0.813      0.858      0.491
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/200      10.7G      1.214     0.6942      1.128          6        640: 100%|██████████| 13/13 [00:03<00:00,  3.34it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.45it/s]
                   all         44         45      0.868      0.878      0.873      0.542
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/200      10.7G      1.237     0.7292      1.134          9        640: 100%|██████████| 13/13 [00:03<00:00,  3.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.13it/s]
                   all         44         45      0.874      0.867       0.87      0.505
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/200      10.7G       1.24     0.7388      1.136          9        640: 100%|██████████| 13/13 [00:04<00:00,  3.16it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.40it/s]
                   all         44         45      0.897      0.867      0.865      0.507
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/200      10.7G      1.258     0.7351      1.129         11        640: 100%|██████████| 13/13 [00:04<00:00,  3.18it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.15it/s]
                   all         44         45      0.914      0.778      0.855        0.5
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/200      10.7G      1.185      0.716      1.072         12        640: 100%|██████████| 13/13 [00:03<00:00,  3.31it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.23it/s]
                   all         44         45      0.894      0.822      0.827      0.543
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/200      10.7G      1.191     0.7244      1.088         11        640: 100%|██████████| 13/13 [00:03<00:00,  3.33it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.45it/s]
                   all         44         45      0.843      0.867      0.871       0.56
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/200      10.7G      1.214     0.7467      1.098         17        640: 100%|██████████| 13/13 [00:03<00:00,  3.32it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.65it/s]
                   all         44         45      0.875      0.844      0.853      0.497
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/200      10.7G      1.151     0.6733      1.099          8        640: 100%|██████████| 13/13 [00:03<00:00,  3.42it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.63it/s]
                   all         44         45      0.901      0.806      0.865      0.616
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/200      10.7G      1.194     0.6985      1.119          7        640: 100%|██████████| 13/13 [00:03<00:00,  3.28it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.85it/s]
                   all         44         45      0.971      0.752      0.848      0.549
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/200      10.7G      1.166     0.7081      1.106         13        640: 100%|██████████| 13/13 [00:04<00:00,  3.19it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.13it/s]
                   all         44         45      0.856      0.795      0.856      0.562
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/200      10.7G      1.187     0.7165      1.121         13        640: 100%|██████████| 13/13 [00:03<00:00,  3.41it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.33it/s]
                   all         44         45      0.839      0.844      0.842      0.549
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/200      10.7G      1.158     0.6939      1.098         11        640: 100%|██████████| 13/13 [00:04<00:00,  3.23it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.09it/s]
                   all         44         45      0.876      0.785      0.856      0.555
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/200      10.7G      1.162      0.684      1.096         13        640: 100%|██████████| 13/13 [00:04<00:00,  3.23it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.38it/s]
                   all         44         45      0.871      0.752       0.83      0.522
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/200      10.7G      1.133       0.66      1.074          5        640: 100%|██████████| 13/13 [00:03<00:00,  3.28it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.04it/s]
                   all         44         45      0.836      0.844      0.856      0.575
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/200      10.7G      1.174     0.7065      1.085          8        640: 100%|██████████| 13/13 [00:04<00:00,  3.22it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.14it/s]
                   all         44         45      0.838        0.8      0.864      0.452
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/200      10.7G      1.159     0.6955      1.104         11        640: 100%|██████████| 13/13 [00:04<00:00,  3.19it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.49it/s]
                   all         44         45      0.904      0.837      0.872      0.545
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/200      10.7G      1.165     0.6983      1.096         15        640: 100%|██████████| 13/13 [00:03<00:00,  3.42it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.99it/s]
                   all         44         45      0.899      0.822       0.85       0.51
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/200      10.7G      1.123     0.6476      1.078         15        640: 100%|██████████| 13/13 [00:03<00:00,  3.45it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.06it/s]
                   all         44         45      0.949      0.822      0.885       0.56
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/200      10.7G       1.16     0.6892      1.104         13        640: 100%|██████████| 13/13 [00:04<00:00,  3.15it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.13it/s]
                   all         44         45      0.913        0.8      0.846      0.555
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/200      10.7G      1.187     0.6919      1.114          5        640: 100%|██████████| 13/13 [00:03<00:00,  3.30it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.01it/s]
                   all         44         45      0.839      0.867      0.853      0.491
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/200      10.7G      1.174     0.6649      1.088         10        640: 100%|██████████| 13/13 [00:03<00:00,  3.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.16it/s]
                   all         44         45       0.86      0.822      0.851      0.505
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/200      10.7G       1.14     0.6355      1.079          6        640: 100%|██████████| 13/13 [00:03<00:00,  3.39it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.37it/s]
                   all         44         45      0.883      0.778      0.837      0.538
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/200      10.7G      1.134     0.6417      1.095          7        640: 100%|██████████| 13/13 [00:03<00:00,  3.29it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.47it/s]
                   all         44         45      0.823      0.826      0.835      0.545
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/200      10.7G      1.092     0.6217      1.065         12        640: 100%|██████████| 13/13 [00:03<00:00,  3.32it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.32it/s]
                   all         44         45      0.865      0.756      0.765      0.484
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/200      10.7G      1.081     0.6515      1.043         12        640: 100%|██████████| 13/13 [00:04<00:00,  3.21it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.40it/s]
                   all         44         45      0.789      0.756      0.792      0.508
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/200      10.7G      1.125     0.6875      1.084         13        640: 100%|██████████| 13/13 [00:03<00:00,  3.45it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  3.44it/s]
                   all         44         45      0.869       0.74       0.84      0.558
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/200      10.7G      1.102     0.6185      1.067         15        640: 100%|██████████| 13/13 [00:03<00:00,  3.35it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.97it/s]
                   all         44         45      0.839      0.811       0.86      0.559
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/200      10.7G       1.09     0.6349      1.056         13        640: 100%|██████████| 13/13 [00:03<00:00,  3.45it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  2.78it/s]
                   all         44         45       0.79      0.867      0.843       0.57
EarlyStopping: Training stopped early as no improvement observed in last 20 epochs. Best results observed at epoch 53, best model saved as best.pt.
To update EarlyStopping(patience=20) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.
73 epochs completed in 0.105 hours.
Optimizer stripped from runs/detect/train8/weights/last.pt, 5.5MB
Optimizer stripped from runs/detect/train8/weights/best.pt, 5.5MB

Validating runs/detect/train8/weights/best.pt...
Ultralytics 8.3.134 🚀 Python-3.11.11 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)
YOLO11n summary (fused): 100 layers, 2,582,347 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  1.95it/s]
                   all         44         45       0.85      0.889      0.882      0.605
/usr/local/lib/python3.11/dist-packages/matplotlib/colors.py:721: RuntimeWarning: invalid value encountered in less
  xa[xa < 0] = -1
/usr/local/lib/python3.11/dist-packages/matplotlib/colors.py:721: RuntimeWarning: invalid value encountered in less
  xa[xa < 0] = -1
Speed: 0.2ms preprocess, 6.8ms inference, 0.0ms loss, 1.3ms postprocess per image
Results saved to runs/detect/train8
License Plate Detection and OCR Documentation
Overview
This script performs license plate detection and Optical Character Recognition (OCR) on a set of images. It uses the YOLO object detection model to locate license plates and Tesseract OCR to extract text from the detected plates. The script is designed for Turkish license plates and includes preprocessing steps to improve OCR accuracy.

Workflow
Image Loading:

Randomly selects a subset of images from the dataset.
Reads and converts images to RGB format for processing.
License Plate Detection:

Uses a YOLO model to detect license plates in the images.
Extracts bounding box coordinates for each detected plate.
Preprocessing:

Converts the license plate region to grayscale.
Crops the left side of the plate (removing the blue "TR" section).
Applies Gaussian blur and thresholding to enhance text clarity.
OCR:

Uses Tesseract OCR to extract text from the preprocessed license plate region.
Formats the extracted text to match Turkish license plate standards.
Visualization:

Draws bounding boxes and recognized text on the images.
Displays the processed images with Matplotlib.
Output:

Prints the recognized license plate text for each image.
Key Functions and Code Snippets
YOLO Model Initialization
model = YOLO("runs/detect/train/weights/best.pt")
Loads the YOLO model with pre-trained weights.
Used for detecting license plates in images.
OCR Preprocessing
plate_img = img[y1:y2, x1:x2]
gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
cut_x = int(w * 0.10)  # Remove 10% from the left
gray = gray[:, cut_x:]
blurred = cv2.GaussianBlur(gray, (5, 5), 1)
_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
Extracts the license plate region from the image.
Converts the region to grayscale and removes the left side.
Applies Gaussian blur and thresholding to prepare the region for OCR.
OCR with Tesseract
text = pytesseract.image_to_string(thresh, config=custom_config)
formatted_text = format_turkish_plate(text)
Extracts text from the preprocessed license plate region.
Formats the text to match Turkish license plate standards.
Visualization
cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)
cv2.putText(img_rgb, formatted_text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
Draws a bounding box around the detected license plate.
Adds the recognized text above the bounding box.
Displaying Results
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.axis("off")
plt.show()
Displays the processed image with bounding boxes and recognized text.
Example Usage
# Load images
image_paths = glob("/path/to/images/*.jpg")
random_images = random.sample(image_paths, 10)

# Process each image
for img_path in random_images:
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_path, conf=0.4, iou=0.4)
    # Further processing as described above
Notes
Ensure that the YOLO model weights and Tesseract OCR are correctly installed and configured.
The script is optimized for Turkish license plates but can be adapted for other formats with minor modifications.
Adjust the confidence (conf) and IoU thresholds for YOLO as needed.
Output Example
For an image with a detected license plate, the script outputs:

The processed image with bounding boxes and recognized text.
The recognized license plate text printed to the console:
Plate 1: 34ABC123
import cv2
import random
import pytesseract
import matplotlib.pyplot as plt
from glob import glob
from ultralytics import YOLO

# Tesseract ayarları
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# YOLO modelini yükle
model = YOLO("runs/detect/train/weights/best.pt")

# Görüntüleri al
image_paths = glob("/kaggle/input/turkish-license-plate-dataset/images/*.png") + \
              glob("/kaggle/input/turkish-license-plate-dataset/images/*.jpg")

num_images = min(10, len(image_paths))
random_images = random.sample(image_paths, num_images)

# Boşluksuz Türk plaka formatlayıcı
def format_turkish_plate(text):
    return text.strip().upper().replace(" ", "").replace("-", "")

# Görüntüler üzerinde işlem yap
for img_path in random_images:
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img_path, conf=0.4, iou=0.4)
    detected_text_list = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()

            # Plaka bölgesini al ve ön işleme uygula
            plate_img = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

            # Sol taraf (mavi TR kısmı) kesiliyor
            h, w = gray.shape
            cut_x = int(w * 0.10)  # %20'sini kes
            gray = gray[:, cut_x:]

            # Orta düzey Gaussian Blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 1)

            # Threshold ile netleştirme
            _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR işlemi
            text = pytesseract.image_to_string(thresh, config=custom_config)
            formatted_text = format_turkish_plate(text)

            # Eğer plaka uzunluğu yeterliyse kutu çiz ve yazıyı ekle
            if len(formatted_text) >= 6:
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(img_rgb, formatted_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                detected_text_list.append(formatted_text)

    # Görüntüyü göster
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

    # Tanınan plakaları yazdır
    for i, text in enumerate(detected_text_list):
        print(f"Plate {i + 1}: {text}")
image 1/1 /kaggle/input/turkish-license-plate-dataset/images/89.jpg: 640x384 1 number_plate, 9.9ms
Speed: 2.3ms preprocess, 9.9ms inference, 1.7ms postprocess per image at shape (1, 3, 640, 384)

image 1/1 /kaggle/input/turkish-license-plate-dataset/images/67.jpg: 640x384 1 number_plate, 9.2ms
Speed: 2.4ms preprocess, 9.2ms inference, 1.5ms postprocess per image at shape (1, 3, 640, 384)

image 1/1 /kaggle/input/turkish-license-plate-dataset/images/546.jpg: 640x640 1 number_plate, 9.6ms
Speed: 3.6ms preprocess, 9.6ms inference, 1.5ms postprocess per image at shape (1, 3, 640, 640)

Plate 1: 66LN430

image 1/1 /kaggle/input/turkish-license-plate-dataset/images/1852.jpg: 640x480 1 number_plate, 10.1ms
Speed: 2.1ms preprocess, 10.1ms inference, 1.5ms postprocess per image at shape (1, 3, 640, 480)

Plate 1: 06HLA95

image 1/1 /kaggle/input/turkish-license-plate-dataset/images/1837.jpg: 640x480 1 number_plate, 9.8ms
Speed: 2.3ms preprocess, 9.8ms inference, 1.5ms postprocess per image at shape (1, 3, 640, 480)

Plate 1: 34H65B33

image 1/1 /kaggle/input/turkish-license-plate-dataset/images/791.jpg: 640x640 2 number_plates, 9.9ms
Speed: 3.6ms preprocess, 9.9ms inference, 1.4ms postprocess per image at shape (1, 3, 640, 640)

Plate 1: 06FU1037
Plate 2: 06BRP867

image 1/1 /kaggle/input/turkish-license-plate-dataset/images/1176.jpg: 640x480 2 number_plates, 9.5ms
Speed: 2.3ms preprocess, 9.5ms inference, 1.4ms postprocess per image at shape (1, 3, 640, 480)

Plate 1: 66AT263
Plate 2: ROLETLELTRDR

image 1/1 /kaggle/input/turkish-license-plate-dataset/images/1030.jpg: 640x512 2 number_plates, 43.2ms
Speed: 2.1ms preprocess, 43.2ms inference, 1.4ms postprocess per image at shape (1, 3, 640, 512)

Plate 1: 06AH7056

image 1/1 /kaggle/input/turkish-license-plate-dataset/images/1128.jpg: 480x640 1 number_plate, 9.8ms
Speed: 2.5ms preprocess, 9.8ms inference, 1.6ms postprocess per image at shape (1, 3, 480, 640)

Plate 1: 34BFT797

image 1/1 /kaggle/input/turkish-license-plate-dataset/images/692.jpg: 640x640 3 number_plates, 10.2ms
Speed: 3.7ms preprocess, 10.2ms inference, 1.5ms postprocess per image at shape (1, 3, 640, 640)

Plate 1: 66LB190