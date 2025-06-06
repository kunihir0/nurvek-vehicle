import os
import glob
from xml.dom import minidom
import random
import shutil
import yaml
from ultralytics import YOLO
import torch
import argparse
import sys

# --- Configuration ---
BASE_MODEL_PATH = "src/models/yolo/yolo11n.pt"  # Pretrained nano model
IMAGES_INPUT_DIR = "data/lplates_imgs"
XML_ANNOTATIONS_INPUT_DIR = "data/lplates_annotations"

# Intermediate and final dataset directories
YOLO_LABELS_CONVERTED_DIR = "data/lplates_yolo_labels_converted"
FINAL_DATASET_BASE_DIR = "data/lp_yolo_dataset_for_training"
TRAIN_IMAGES_DIR = os.path.join(FINAL_DATASET_BASE_DIR, "images", "train")
VAL_IMAGES_DIR = os.path.join(FINAL_DATASET_BASE_DIR, "images", "val")
TRAIN_LABELS_DIR = os.path.join(FINAL_DATASET_BASE_DIR, "labels", "train")
VAL_LABELS_DIR = os.path.join(FINAL_DATASET_BASE_DIR, "labels", "val")
DATA_YAML_PATH = os.path.join(FINAL_DATASET_BASE_DIR, "lp_data.yaml")

# Dataset parameters
CLASS_NAME_IN_XML = "licence"
LICENSE_PLATE_CLASS_ID = 0
VAL_SPLIT_SIZE = 0.2

# Training parameters
TRAINING_EPOCHS = 55
IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = 12
PATIENCE = 20

# --- MODIFIED: Added Colors class for better terminal output ---
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# --- Helper Functions ---

def convert_coordinates(size, box):
    """Converts (xmin, xmax, ymin, ymax) box to YOLO format."""
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
        print(f"{Colors.WARNING}Warning: No XML files found in {xml_dir}{Colors.ENDC}")
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
                        
                        b = (xmin, xmax, ymin, ymax) 
                        bb = convert_coordinates((width, height), b)
                        
                        f_out.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')
                        found_objects_in_file = True
                
                if found_objects_in_file:
                    converted_count +=1

        except Exception as e:
            print(f"{Colors.FAIL}Error processing {fname}: {e}{Colors.ENDC}")
    print(f"XML to YOLO conversion finished. {converted_count} files with '{class_name_in_xml}' objects processed.")

def create_yolo_dataset_directories(base_dir, train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
    """Creates the necessary directory structure for the YOLO dataset."""
    print("Creating dataset directories...")
    paths_to_create = [base_dir, train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]
    for path in paths_to_create:
        os.makedirs(path, exist_ok=True)
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
        print(f"{Colors.FAIL}Error: No images found in {source_images_dir}. Please check the path.{Colors.ENDC}")
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
                print(f"{Colors.WARNING}Warning: Label file not found for image {img_path}. Skipping.{Colors.ENDC}")
        print(f"Copied {copied_count} images and labels to {set_name} set.")

    print(f"Total images: {len(all_image_files)}, Training: {len(train_files)}, Validation: {len(val_files)}")
    copy_files(train_files, train_images_dest, train_labels_dest, "training")
    copy_files(val_files, val_images_dest, val_labels_dest, "validation")
    print("Dataset splitting finished.")

def generate_data_yaml_file(yaml_file_path, base_path, train_images_rel, val_images_rel, class_id, class_name_str):
    """Generates the data.yaml file for YOLO training."""
    print(f"Generating data YAML file at '{yaml_file_path}'...")
    
    data_config = {
        'path': os.path.abspath(base_path),
        'train': train_images_rel, # 'images/train'
        'val': val_images_rel,     # 'images/val'
        'nc': 1,
        'names': {class_id: class_name_str}
    }

    with open(yaml_file_path, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False, default_flow_style=None)
    print(f"Data YAML file generated: {yaml_file_path}")
    print("YAML content:")
    print(yaml.dump(data_config, sort_keys=False, default_flow_style=None))


if __name__ == "__main__":
    print("--- Starting License Plate Detector Training Pipeline ---")

    # Pipeline steps 1-3
    create_yolo_dataset_directories(FINAL_DATASET_BASE_DIR, TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_LABELS_DIR)
    convert_xml_annotations_to_yolo(XML_ANNOTATIONS_INPUT_DIR, YOLO_LABELS_CONVERTED_DIR, CLASS_NAME_IN_XML, LICENSE_PLATE_CLASS_ID)
    split_source_to_train_val(IMAGES_INPUT_DIR, YOLO_LABELS_CONVERTED_DIR,
                              TRAIN_IMAGES_DIR, VAL_IMAGES_DIR,
                              TRAIN_LABELS_DIR, VAL_LABELS_DIR,
                              VAL_SPLIT_SIZE)
    
    # 4. Create data.yaml with relative paths for better portability
    train_rel_path = os.path.relpath(TRAIN_IMAGES_DIR, FINAL_DATASET_BASE_DIR)
    val_rel_path = os.path.relpath(VAL_IMAGES_DIR, FINAL_DATASET_BASE_DIR)
    generate_data_yaml_file(DATA_YAML_PATH, FINAL_DATASET_BASE_DIR, train_rel_path, val_rel_path, LICENSE_PLATE_CLASS_ID, CLASS_NAME_IN_XML)

    # 5. Train the model
    print("\n--- Starting YOLO Model Training ---")
    
    parser = argparse.ArgumentParser(description="YOLOv8 License Plate Detector Training Script")
    parser.add_argument('--device', type=str, default=None, help="Device to use for training, e.g., '0', '1', 'cpu'. Skips interactive selection.")
    args = parser.parse_args()

    device_arg = args.device

    if device_arg is None:
        # --- MODIFIED: Added environment diagnostics and graceful exit ---
        print(f"\n{Colors.HEADER}--- Hardware-Detection ---{Colors.ENDC}")
        # Check for AMD GPUs via ROCm/HIP
        is_rocm_available = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if is_rocm_available:
            print(f"{Colors.OKGREEN}ROCm/HIP backend detected. PyTorch should see AMD GPUs.{Colors.ENDC}")
        
        if torch.cuda.is_available():
            print(f"{Colors.OKCYAN}--- Interactive GPU Selection ---{Colors.ENDC}")
            gpu_count = torch.cuda.device_count()
            print(f"{Colors.BOLD}Available Devices:{Colors.ENDC}")
            for i in range(gpu_count):
                print(f"  {Colors.OKGREEN}[{i}]{Colors.ENDC} {torch.cuda.get_device_name(i)}")
            print(f"  {Colors.WARNING}[{gpu_count}]{Colors.ENDC} CPU")

            try:
                while True:
                    try:
                        choice_str = input(f"Enter your choice (0-{gpu_count}): ")
                        if not choice_str:
                            print(f"{Colors.FAIL}No selection made. Please enter a number.{Colors.ENDC}")
                            continue
                        choice = int(choice_str)
                        if 0 <= choice < gpu_count:
                            device_arg = str(choice)
                            break
                        elif choice == gpu_count:
                            device_arg = 'cpu'
                            break
                        else:
                            print(f"{Colors.FAIL}Invalid choice. Please select a number between 0 and {gpu_count}.{Colors.ENDC}")
                    except ValueError:
                        print(f"{Colors.FAIL}Invalid input. Please enter a number.{Colors.ENDC}")
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}Operation cancelled by user. Exiting.{Colors.ENDC}")
                sys.exit(0)
            print(f"{Colors.OKCYAN}-----------------------------{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}No CUDA/ROCm-enabled GPU found by PyTorch. Using CPU for training.{Colors.ENDC}")
            device_arg = 'cpu'
    
    print(f"\n{Colors.OKGREEN}Selected device for training: {device_arg}{Colors.ENDC}")

    try:
        model = YOLO(BASE_MODEL_PATH)
        
        if not os.path.exists(DATA_YAML_PATH):
            print(f"{Colors.FAIL}ERROR: Data YAML file not found at {DATA_YAML_PATH}. Aborting training.{Colors.ENDC}")
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
                device=device_arg,
                patience=PATIENCE,
                augment=True,
                amp=True,
                project=FINAL_DATASET_BASE_DIR,
                name='train_lp_run',
                exist_ok=True
            )
            print(f"\n{Colors.OKGREEN}--- YOLO Model Training Finished ---{Colors.ENDC}")
            print(f"Training results saved in: {results.save_dir}")
            print(f"Best model saved as: {os.path.join(results.save_dir, 'weights', 'best.pt')}")

    except Exception as e:
        print(f"\n{Colors.FAIL}An error occurred during training: {e}{Colors.ENDC}")

    print(f"\n{Colors.HEADER}--- License Plate Detector Training Pipeline Complete ---{Colors.ENDC}")
