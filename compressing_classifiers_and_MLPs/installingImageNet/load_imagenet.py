import os
from datasets import load_dataset
from PIL import Image
import shutil
from tqdm import tqdm
from huggingface_hub import login
from index_to_synset import create_index_to_synset_dict, load_imagenet_mapping
from pathlib import Path
import warnings
import tomllib

def download_and_organize_imagenet(json_file, output_dir="./imagenet", hf_token=None):
    """
    Download ImageNet-1K and organize it in the standard structure:
    imagenet/
    ├── train/
    │   ├── n01440764/  # synset folders
    │   ├── n01443537/
    │   └── ...
    └── val/
        ├── n01440764/
        ├── n01443537/
        └── ...
    """
    
    os.makedirs(output_dir, exist_ok=True)

    # Create directory structure
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    # test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    # os.makedirs(test_dir, exist_ok=True)

    mapping = load_imagenet_mapping(json_file)
    index_to_synset_dict = create_index_to_synset_dict(mapping)
    
    for label in range(1000):
        synset_name = index_to_synset_dict[label]
        os.makedirs(os.path.join(train_dir, synset_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, synset_name), exist_ok=True)
        # os.makedirs(os.path.join(test_dir, synset_name), exist_ok=True)

    # Authenticate with Hugging Face
    if hf_token:
        login(token=hf_token)
    else:
        # Interactive login (will prompt for token)
        login()
    
    print("Loading ImageNet dataset from Hugging Face...")
    # Try the official ILSVRC dataset first
    try:
        dataset = load_dataset("ILSVRC/imagenet-1k", download_mode="reuse_cache_if_exists")
    except:
        # Fallback to alternative
        print("Trying alternative dataset...")
        dataset = load_dataset("imagenet-1k", download_mode="reuse_cache_if_exists")
    
    # Keep track of skipped images
    skipped_images = []
    already_saved = 0
    newly_saved = 0
    
    def needs_conversion(image):
        """Check if image needs RGB conversion."""
        return image.mode not in ('RGB', 'L')
    
    def save_image_safely(image, image_path, split_name, index):
        """Safely save image with error handling for corrupted files and RGBA mode."""
        try:
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                # Create a white background
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
                image = rgb_image
            elif image.mode not in ('RGB', 'L'):
                # Convert other modes (e.g., 'P', 'LA') to RGB
                image = image.convert('RGB')
            
            # Save the image
            image.save(image_path, "JPEG")
            return True
        except Exception as e:
            print(f"WARNING: Skipping {split_name} image {index} due to error: {str(e)}")
            skipped_images.append((split_name, index, str(e)))
            return False
    
    # Process training set
    print("Organizing training set...")
    for i, sample in enumerate(dataset['train']):
        if i % 10000 == 0:
            print(f"train {i} - saved: {newly_saved}, skipped (already exists): {already_saved}")
        
        image = sample['image']
        label = sample['label']
        synset_name = index_to_synset_dict[label]
        
        image_path = os.path.join(train_dir, synset_name, synset_name+"_"+str(i)+".JPEG")
        
        # Check if file already exists and doesn't need conversion
        if os.path.exists(image_path) and not needs_conversion(image):
            already_saved += 1
            continue
        
        # If file exists but needs conversion, we'll overwrite it
        if os.path.exists(image_path) and needs_conversion(image):
            print(f"Reconverting train image {i} (mode: {image.mode})")
        
        if save_image_safely(image, image_path, "train", i):
            newly_saved += 1
    
    # Process validation set
    print("Organizing validation set...")
    for i, sample in enumerate(dataset['validation']):
        if i % 1000 == 0:
            print(f"val {i} - saved: {newly_saved}, skipped (already exists): {already_saved}")
        
        image = sample['image']
        label = sample['label']
        synset_name = index_to_synset_dict[label]
        
        image_path = os.path.join(val_dir, synset_name, synset_name+"_"+str(i)+".JPEG")
        
        # Check if file already exists and doesn't need conversion
        if os.path.exists(image_path) and not needs_conversion(image):
            already_saved += 1
            continue
        
        # If file exists but needs conversion, we'll overwrite it
        if os.path.exists(image_path) and needs_conversion(image):
            print(f"Reconverting validation image {i} (mode: {image.mode})")
        
        if save_image_safely(image, image_path, "validation", i):
            newly_saved += 1

    print(f"\nDataset organized in: {output_dir}")
    print(f"Structure: {output_dir}/{{train,val}}/{{synset_folders}}/{{images}}")
    print(f"\nStatistics:")
    print(f"  Newly saved: {newly_saved}")
    print(f"  Already existed: {already_saved}")
    print(f"  Skipped (errors): {len(skipped_images)}")
    
    # Save log of skipped images
    if skipped_images:
        log_path = os.path.join(output_dir, "skipped_images.txt")
        with open(log_path, 'w') as f:
            for split, idx, error in skipped_images:
                f.write(f"{split},{idx},{error}\n")
        print(f"Skipped images logged to: {log_path}")

if __name__ == "__main__":
    # Assume script runs from desired working directory
    script_dir = Path.cwd()

    # Load config.toml from parent directory
    config_path = script_dir.parent / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}. Are you running the script from the correct path?")

    with open(config_path, mode="rb") as fp:
        config = tomllib.load(fp)
    hf_token = config["huggingface"]["token"]
    output_directory = config["paths"]["imagenet_path"]

    json_file = script_dir / "imagenet_class_index.json"

    download_and_organize_imagenet(json_file, output_directory, hf_token)