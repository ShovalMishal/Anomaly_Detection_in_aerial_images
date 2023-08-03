# Anomaly Detection in aerial images
Using this repo current version, you can create an anomaly detection experiment where the normal label is background and the abnormal is foreground/OOD.
## Requirements
You should create repositories folder containing this repo, mmrotate and mmdet repos.
Before starting you need to download DOTA-v2 dataset - you can use the following link https://captain-whu.github.io/DOTA/
After, you should create environment using the requirements.txt file.
Next, you need to run the preprocess of DOTA dataset based on https://github.com/ShovalMishal/mmrotate/blob/main/tools/data/dota/README.md, when I used the single scale preprocess.

## Preprocess Data
First, we need to create the image pyramid patches dataset.
For this manner you should run the following command:
python create_image_pyramid_patches_data.py -c <./Anomaly_Detection_in_aerial_images/config.py> -d <./data/patches_dataset/> --dataset_type <subtrain/subval>
There are several flags for this command:
* ---config The relative path to the cfg file
* --dataset_dir The saved dataset path
* --pyramid_levels Number of pyramid levels
* --scale_factor scale factor between pyramid levels
* --patch_stride Stride between patches in %
* --patch_size The patch size 
* --dataset_type shpuld contain subtrain or subval

- please note you have updated the config.py with the right paths!!

## Performing Anomaly detection experiment
Run the following command:
python Anomaly_detection_BG_FG_patches_ver.py -c <./Anomaly_Detection_in_aerial_images/config.py> -o <output_dir> -k <k_values with subspaces>
Then, you create the features dictionary and finally get the test results (Roc curve and precision-recall curve)
There are several flags for this command:
* --config The relative path to the cfg file
* --output_dir Statistics output dir
* --sampled_ratio Sampled ratio for the features dictionary
* -k_values Nearest Neighbours count values
* -use-cached If flagged, use the cached features. Otherwise, recalculate it every time you run the script.