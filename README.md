# Anomaly Detection in aerial images
Using this repo current version, you can run the full OOD detection pipeline.
First, it creates an anomaly detection experiment where the normal label is background and the abnormal is foreground/OOD.
Then, it trains the classifier on the anomaly detection results.

## Requirements
You should create repositories folder containing this repo, mmrotate and mmdet repos.
Before starting you need to download DOTA-v2 dataset - you can use the following link https://captain-whu.github.io/DOTA/
After, you should create environment using the requirements.txt file.
Next, you need to run the preprocess of DOTA dataset based on https://github.com/ShovalMishal/mmrotate/blob/main/tools/data/dota/README.md, when I used the single scale preprocess.

## Create patches dataset
First, we need to create the image pyramid patches dataset.
For this manner you should run the following command:
python create_image_pyramid_patches_data.py -c <./Anomaly_Detection_in_aerial_images/config.py> -d <./data/patches_dataset/> --dataset_type <subtrain/subval/subtest>
There are several flags for this command:
* ---config The relative path to the cfg file
* --dataset_dir The saved dataset path
* --pyramid_levels Number of pyramid levels
* --scale_factor scale factor between pyramid levels
* --patch_stride Stride between patches in %
* --patch_size The patch size 
* --dataset_type shpuld contain subtrain or subval

- please note you have updated the config.py with the right paths!!

## Performing OOD experiment
Run the following command:
python FullOODPipeline.py -c ./Anomaly_Detection_in_aerial_images/config.py -o ./Anomaly_Detection_in_aerial_images/out
Then, you create the features dictionary, inclusion files (files with samples we use in next stage - OOD detection), 
labels and scores files (to calculate those only once) and finally get the test results (Roc curve and precision-recall
curve) for the anomaly detetction stage. After, the classifier is fine-tuned for the OOD stage. Finally, you get the
statistics for the OOD detection experiment.
There are several flags for this command:
* --config The relative path to the cfg file
* --output_dir All files are saved in this path.


# Internal documentation:
To run this on our lab's A5000, we go to:
```shell
cd ~/Documents/Repositories/
```

Then, we type:
```shell
python Anomaly_Detection_in_aerial_images/FullOODPipeline.py -c ./Anomaly_Detection_in_aerial_images/config.py -o ./Anomaly_Detection_in_aerial_images/out
```

