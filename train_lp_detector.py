import os
import glob
from xml.dom import minidom
import random
import shutil
import yaml
from ultralytics import YOLO
import torch

# --- Configuration ---
BASE_MODEL_PATH = "src/models/yolo/yolo11n.pt"  # Pretrained nano model
IMAGES_INPUT_DIR = "data/lplates_imgs"
XML_ANNOTATIONS_INPUT_DIR = "data/lplates_annotations"

# Intermediate and final dataset directories
YOLO_LABELS_CONVERTED_DIR = "data/lplates_yolo_labels_converted"  # For YOLO format labels from XML
FINAL_DATASET_BASE_DIR = "data/lp_yolo_dataset_for_training"
TRAIN_IMAGES_DIR = os.path.join(FINAL_DATASET_BASE_DIR, "images", "train")
VAL_IMAGES_DIR = os.path.join(FINAL_DATASET_BASE_DIR, "images", "val")
TRAIN_LABELS_DIR = os.path.join(FINAL_DATASET_BASE_DIR, "labels", "train")
VAL_LABELS_DIR = os.path.join(FINAL_DATASET_BASE_DIR, "labels", "val")
DATA_YAML_PATH = os.path.join(FINAL_DATASET_BASE_DIR, "lp_data.yaml")

# Dataset parameters
CLASS_NAME_IN_XML = "licence" # Corrected based on Cars6.xml
LICENSE_PLATE_CLASS_ID = 0
VAL_SPLIT_SIZE = 0.2  # 20% for validation

# Training parameters
TRAINING_EPOCHS = 100 # Increased for longer training
IMAGE_SIZE = 640      # Increased image size due to more VRAM
BATCH_SIZE = 32       # Increased batch size due to more VRAM
WORKERS = 2           # Increased workers
PATIENCE = 20         # Increased patience for early stopping

# --- Helper Functions ---

def convert_coordinates(size, box):
    """Converts (xmin, xmax, ymin, ymax) box to YOLO format (x_center, y_center, width, height) normalized."""
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

def convert_xml_annotations_to_yolo(xml_dir, yolo_labels_dir, class_name_in_xml, target_class_id):
    """Converts XML annotations to YOLO format."""
    print(f"Starting XML to YOLO conversion from '{xml_dir}' to '{yolo_labels_dir}'...")
    os.makedirs(yolo_labels_dir, exist_ok=True)
    
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    if not xml_files:
        print(f"Warning: No XML files found in {xml_dir}")
        return

    converted_count = 0
    for fname in xml_files:
        try:
            xmldoc = minidom.parse(fname)
            annot_fname_base = os.path.splitext(os.path.basename(fname))[0]
            yolo_output_filepath = os.path.join(yolo_labels_dir, f"{annot_fname_base}.txt")

            with open(yolo_output_filepath, "w") as f_out:
                itemlist = xmldoc.getElementsByTagName('object')
                size_node = xmldoc.getElementsByTagName('size')[0]
                width = int(size_node.getElementsByTagName('width')[0].firstChild.data)
                height = int(size_node.getElementsByTagName('height')[0].firstChild.data)

                found_objects_in_file = False
                for item in itemlist:
                    class_name_xml = item.getElementsByTagName('name')[0].firstChild.data
                    if class_name_xml == class_name_in_xml:
                        label_str = str(target_class_id)
                        
                        bndbox_node = item.getElementsByTagName('bndbox')[0]
                        xmin = float(bndbox_node.getElementsByTagName('xmin')[0].firstChild.data)
                        ymin = float(bndbox_node.getElementsByTagName('ymin')[0].firstChild.data)
                        xmax = float(bndbox_node.getElementsByTagName('xmax')[0].firstChild.data)
                        ymax = float(bndbox_node.getElementsByTagName('ymax')[0].firstChild.data)
                        
                        # XML format is often (xmin, ymin, xmax, ymax)
                        # convert_coordinates expects (xmin, xmax, ymin, ymax)
                        b = (xmin, xmax, ymin, ymax) 
                        bb = convert_coordinates((width, height), b)
                        
                        f_out.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')
                        found_objects_in_file = True
                
                if found_objects_in_file:
                    converted_count +=1
                # else:
                    # print(f"Note: No objects of class '{class_name_in_xml}' found in {fname}, or file was empty.")

        except Exception as e:
            print(f"Error processing {fname}: {e}")
    print(f"XML to YOLO conversion finished. {converted_count} files with '{class_name_in_xml}' objects processed.")

def create_yolo_dataset_directories(base_dir, train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
    """Creates the necessary directory structure for the YOLO dataset."""
    print("Creating dataset directories...")
    paths_to_create = [base_dir, train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]
    for path in paths_to_create:
        os.makedirs(path, exist_ok=True)
        # print(f"Ensured directory exists: {path}")
    print("Dataset directories created/ensured.")

def split_source_to_train_val(source_images_dir, source_yolo_labels_dir,
                              train_images_dest, val_images_dest,
                              train_labels_dest, val_labels_dest,
                              validation_split_ratio):
    """Splits image and label files into training and validation sets."""
    print(f"Splitting dataset from '{source_images_dir}' and '{source_yolo_labels_dir}'...")
    
    image_extensions = ("*.jpg", "*.jpeg", "*.png")
    all_image_files = []
    for ext in image_extensions:
        all_image_files.extend(glob.glob(os.path.join(source_images_dir, ext)))

    if not all_image_files:
        print(f"Error: No images found in {source_images_dir}. Please check the path and image extensions.")
        return

    random.shuffle(all_image_files)
    
    num_val_images = int(len(all_image_files) * validation_split_ratio)
    val_files = all_image_files[:num_val_images]
    train_files = all_image_files[num_val_images:]

    def copy_files(file_list, img_dest_folder, lbl_dest_folder, set_name):
        copied_count = 0
        for img_path in file_list:
            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            label_filename = f"{base_filename}.txt"
            source_label_path = os.path.join(source_yolo_labels_dir, label_filename)

            if os.path.exists(source_label_path):
                shutil.copy2(img_path, os.path.join(img_dest_folder, os.path.basename(img_path)))
                shutil.copy2(source_label_path, os.path.join(lbl_dest_folder, label_filename))
                copied_count += 1
            else:
                print(f"Warning: Label file not found for image {img_path} (expected at {source_label_path}). Skipping this image.")
        print(f"Copied {copied_count} images and labels to {set_name} set.")

    print(f"Total images: {len(all_image_files)}, Training: {len(train_files)}, Validation: {len(val_files)}")
    copy_files(train_files, train_images_dest, train_labels_dest, "training")
    copy_files(val_files, val_images_dest, val_labels_dest, "validation")
    print("Dataset splitting finished.")

def generate_data_yaml_file(yaml_file_path, abs_path_to_train_images, abs_path_to_val_images, class_id, class_name_str):
    """Generates the data.yaml file for YOLO training."""
    print(f"Generating data YAML file at '{yaml_file_path}'...")
    
    # Ultralytics prefers absolute paths or paths relative to the yolov5/yolov8 directory if run from there.
    # For simplicity and clarity when running this script from project root, we use absolute paths.
    data_config = {
        'path': os.path.abspath(os.path.dirname(yaml_file_path)), # dataset root dir
        'train': os.path.abspath(abs_path_to_train_images),       # train images (relative to 'path')
        'val': os.path.abspath(abs_path_to_val_images),           # val images (relative to 'path')
        'nc': 1,                                                 # number of classes
        'names': {class_id: class_name_str}                      # class names
    }
    
    # Adjust train/val paths to be relative to the 'path' key if Ultralytics expects that
    # For current Ultralytics versions, absolute paths in train/val are fine.
    # If issues arise, they can be made relative:
    # data_config['train'] = os.path.relpath(abs_path_to_train_images, data_config['path'])
    # data_config['val'] = os.path.relpath(abs_path_to_val_images, data_config['path'])


    with open(yaml_file_path, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False, default_flow_style=None)
    print(f"Data YAML file generated: {yaml_file_path}")
    print("YAML content:")
    print(yaml.dump(data_config, sort_keys=False, default_flow_style=None))


if __name__ == "__main__":
    print("--- Starting License Plate Detector Training Pipeline ---")

    # 1. Create dataset directories
    create_yolo_dataset_directories(FINAL_DATASET_BASE_DIR, TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_LABELS_DIR)

    # 2. Convert XML annotations to YOLO format
    convert_xml_annotations_to_yolo(XML_ANNOTATIONS_INPUT_DIR, YOLO_LABELS_CONVERTED_DIR, CLASS_NAME_IN_XML, LICENSE_PLATE_CLASS_ID)

    # 3. Split dataset into train/val
    split_source_to_train_val(IMAGES_INPUT_DIR, YOLO_LABELS_CONVERTED_DIR,
                              TRAIN_IMAGES_DIR, VAL_IMAGES_DIR,
                              TRAIN_LABELS_DIR, VAL_LABELS_DIR,
                              VAL_SPLIT_SIZE)

    # 4. Create data.yaml
    # Note: YOLOv8 typically expects train/val paths in data.yaml to be relative to the dataset root or absolute.
    # We will provide absolute paths for clarity.
    generate_data_yaml_file(DATA_YAML_PATH, 
                            TRAIN_IMAGES_DIR, # Pass relative path, will be made absolute in function
                            VAL_IMAGES_DIR,   # Pass relative path, will be made absolute in function
                            LICENSE_PLATE_CLASS_ID, 
                            CLASS_NAME_IN_XML) # Use the same name as in XML for consistency

    # 5. Train the model
    print("--- Starting YOLO Model Training ---")
    
    # Check for AMD GPU (ROCm/HIP) first, then CUDA, then CPU
    if hasattr(torch.version, 'hip') and torch.version.hip is not None and torch.cuda.is_available() and torch.cuda.device_count() > 0 and 'amdgpu' in torch.cuda.get_device_name(0).lower():
        # This is a common way to check if ROCm is the backend for cuda.is_available()
        # Or more directly, try to use 'hip'
        try:
            if torch.cuda.device_count() > 0: # ROCm might report AMD GPUs via cuda interface
                 # Check if the first GPU is AMD
                if 'amd' in torch.cuda.get_device_name(0).lower():
                    device_arg = 'hip:0' # Or just 'hip' to let PyTorch pick
                    print(f"Attempting to use AMD GPU via ROCm/HIP: {device_arg}")
                else: # Fallback if first GPU is not AMD, but CUDA is available
                    device_arg = 0
                    print(f"First GPU is not AMD. Using CUDA GPU: {device_arg} ({torch.cuda.get_device_name(0)})")
            else: # Should not happen if torch.cuda.is_available is true
                device_arg = 'cpu'
                print("ROCm/HIP check passed but no GPUs found via CUDA interface. Using CPU.")
        except Exception as e_hip_check:
            print(f"Error during ROCm/HIP device check, falling back: {e_hip_check}")
            if torch.cuda.is_available(): # Fallback to CUDA if HIP check failed
                device_arg = 0
                print(f"Using CUDA GPU: 0 ({torch.cuda.get_device_name(0)})")
            else:
                device_arg = 'cpu'
                print("No GPU available (ROCm/HIP or CUDA). Using CPU.")

    elif torch.cuda.is_available(): # Standard CUDA check if ROCm not detected first
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            device_arg = list(range(gpu_count))
            print(f"Using {gpu_count} CUDA GPUs: {device_arg}")
        else:
            device_arg = 0
            print(f"Using 1 CUDA GPU: 0 ({torch.cuda.get_device_name(0)})")
    else:
        device_arg = 'cpu'
        print("No GPU available (neither ROCm/HIP nor CUDA). Using CPU for training.")
    
    print(f"Selected device for training: {device_arg}")

    try:
        model = YOLO(BASE_MODEL_PATH)  # Load pretrained yolo11n.pt
        
        # Check if DATA_YAML_PATH exists before training
        if not os.path.exists(DATA_YAML_PATH):
            print(f"ERROR: Data YAML file not found at {DATA_YAML_PATH}. Aborting training.")
        else:
            print(f"Training with data configuration: {DATA_YAML_PATH}")
            print(f"Using base model: {BASE_MODEL_PATH}")
            print(f"Training parameters: Epochs={TRAINING_EPOCHS}, ImgSz={IMAGE_SIZE}, Batch={BATCH_SIZE}, Workers={WORKERS}, Patience={PATIENCE}")
            
            results = model.train(
                data=DATA_YAML_PATH,
                epochs=TRAINING_EPOCHS,
                imgsz=IMAGE_SIZE,
                batch=BATCH_SIZE,
                workers=WORKERS,
                device=device_arg, # PyTorch should auto-detect ROCm GPU if env var is set
                patience=PATIENCE,
                augment=True, # Explicitly enable augmentations
                amp=False,  # Explicitly disable Automatic Mixed Precision
                project=FINAL_DATASET_BASE_DIR, # Saves results under FINAL_DATASET_BASE_DIR/
                name='train_lp_run_amd',      # New run name for AMD training
                exist_ok=True                 # Allow overwriting if 'train_lp_run_amd' exists
            )
            print("--- YOLO Model Training Finished ---")
            print(f"Training results saved in: {os.path.join(FINAL_DATASET_BASE_DIR, 'train_lp_run')}")
            print(f"Best model saved as: {os.path.join(FINAL_DATASET_BASE_DIR, 'train_lp_run', 'weights', 'best.pt')}")

    except Exception as e:
        print(f"An error occurred during training: {e}")

    print("--- License Plate Detector Training Pipeline Complete ---")