# Anomaly Detection in aerial images
Using this repo current version, you can create an anomaly detection experiment where the normal label is background and the abnormal is foreground/OOD.
## Requirements
You should create repositories folder containing this repo, mmrotate and mmdet repos.
Before starting you need to download DOTA-v2 dataset - you can use the following link https://captain-whu.github.io/DOTA/
After, you should create environment using the requirements.txt file.
Next, you need to run the preprocess of DOTA dataset based on https://github.com/ShovalMishal/mmrotate/blob/main/tools/data/dota/README.md, when I used the single scale preprocess.
## Performing Anomaly detection experiment
Run the following command:
python Anomaly_detection_BG_FG.py -c <oriented-rcnn-le90_r50_fpn_1x_dota_for_anomaly.py> -o <output_dir> -k <k_values with subspaces>
Then, you create the features dictionary and finally get the test results (Roc curve and precision-recall curve)