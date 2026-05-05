# Installing ImageNet via Hugging Face

Here we provide a simple Python script to **download and organize the ImageNet-1K dataset** using the [🤗 Hugging Face Datasets](https://huggingface.co/docs/datasets) library.
The script will automatically structure the dataset into the conventional ImageNet folder layout for training and validation.

Users are responsible for:
- Obtaining proper access/licenses for datasets they use
- Complying with ImageNet's terms of service
- Ensuring their use case is permitted (academic/research only)

## Prerequisites

- Python 3.11+ is installed and accessible via the `python` command.
- You have a [Hugging Face](https://huggingface.co/) account and access to the ImageNet dataset (which is gated).
- You have [conda](https://docs.conda.io/en/latest/) or another environment manager installed.

## 1. Obtain a Hugging Face Token

1. Log in to your [Hugging Face account](https://huggingface.co/).
2. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens).
3. Create a **new token** with the following permission enabled:
   ```
   Read access to contents of all public gated repos you can access
   ```
4. Copy the generated token

## 2. Configure Paths and Token

Copy the example configuration file and edit it to match your local setup:

```bash
user@machine:~$ cd /path/to/compressing_classifiers_and_MLPs
user@machine:/path/to/compressing_classifiers_and_MLPs$ cp config.toml.example config.toml
```

Then open `config.toml` and update the following values:

```toml
[huggingface]
token = "YOUR_HF_TOKEN_HERE"

[paths]
imagenet_path = "/path/to/folder_containing_the_imagenet_dataset"
```

Replace `YOUR_HF_TOKEN_HERE` with your actual token and specify where the imagenet dataset should be saved by updating `imagenet_path`. 


**Note:** The `config.toml` file is `.gitignored` to keep your token and absolute paths private.

---

## 3. Create and Activate a Python Environment

Navigate to the folder containing this `.md` file and setup a python environment with the required packages:
```bash
(base) user@machine:~$ cd /path/to/installingImageNet
(base) user@machine:/path/to/installingImageNet$ conda create -n imageNet
(base) user@machine:/path/to/installingImageNet$ conda activate imageNet
(imageNet) user@machine:/path/to/installingImageNet$ conda install pip
(imageNet) user@machine:/path/to/installingImageNet$ pip install -r requirements.txt
```

## 4. Download and Organize ImageNet

Once the environment is set up and your token is configured, simply run:

```bash
(imageNet) user@machine:/path/to/installingImageNet$ python load_imagenet.py
```

This script will:

- Authenticate to Hugging Face using your token.
- Download the ImageNet-1K dataset.
- Automatically create a directory structure like:

```
imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

Images are saved in `.JPEG` format, converted as needed to ensure compatibility.

---

## 5. Output and Logs

At the end of the process, you’ll see a summary like:

```
Dataset organized in: /path/to/imagenet
Structure: /path/to/imagenet/{train,val}/{synset_folders}/{images}

Statistics:
  Newly saved: 1281167
  Already existed: 0
  Skipped (errors): 12
```

If any corrupted or invalid images are encountered, they are skipped and logged in:

```
imagenet/skipped_images.txt
```

## Additional Notes

- The script uses the `index_to_synset` mapping provided in `imagenet_class_index.json` to organize images into class-specific folders.
- If the primary Hugging Face dataset (`ILSVRC/imagenet-1k`) is unavailable, the script automatically falls back to an alternative source (`imagenet-1k`).
- The process can be resumed safely; already existing valid files are skipped.

## License and Usage

Please ensure that you have appropriate academic or research access to the ImageNet dataset and comply with its terms of use: [https://image-net.org/download.php](https://image-net.org/download.php)
