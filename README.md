# Anomaly_Detection_in_aerial_images
Using this repo current version, you can create BB datasets from the DOTA datasets, train VIT classifier and eventually based on both, you can create ood mechanism.
## Requirements
Before starting you need to download DOTA-v2 dataset - you can use the following link https://captain-whu.github.io/DOTA/
After, you should create environment using the requirements.txt file.
## Step 1 - cretaing the BB datasets
Run the following command:
python BB_dataset_creator.py -p "./" -t -th 0.2 
and 
python BB_dataset_creator.py -p "./" -th 0.2
where:
* -p is where you need to specify the path for saving the dataset folder.
* -t is a boolian which specify if you create the train or the test set.
* -th is a float which specify the threshold of the blocking BB smallest side.
You should run this script twice - one for test set and once for train set. 
Eventually you get the BB dataset's folder in the specified path and histograms of the train and test sets classes ditributions in the statistics folder.
## Step 2 - training VIT classfier
Run the following command:
python training_model.py -p <path to the train and test set folder> -id "ship" "large-vehicle" "harbor" "tennis-court" "plane" "small-vehicle"
where:
* -p is where you need to specify the path for the input dataset folder.
* -id is where you should specify the list of the in ditribution classes.
## Step 3 - Running the OOD detector:
python naive_OOD_detector.py -p <path to the train and test set folder> -o "./vit-bb-dataset44" -id "ship" "large-vehicle" "harbor" "tennis-court" "plane" "small-vehicle" -od "bridge" "ground-track-field" "soccer-ball-field" "roundabout" "storage-tank" "baseball-diamond" "swimming-pool" "airport" "container-crane" "basketball-court" "helicopter" "helipad" -s 1000
where:
* -p is where you need to specify the path for the input dataset folder.
* -o is where the trained model was saved.
* -id is where you should specify the list of the in ditribution classes.
* -od is where you should specify the list of the ood classes.
* -s is where you should specify the number of training samples for the "KNN" classification task.


