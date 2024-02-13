# Anomaly Detection in aerial images
Using this repo current version, you can run the full OOD detection pipeline.
First, it creates an anomaly detection experiment in order to filter out the objects from each image.
Then, it trains the classifier on the anomaly detection results.
Finally, testing ood model, based on the trained classifier in order to detect ood labels from the aerial images.

## Requirements
You should clone this repo, mmrotate and mmdet repos.

mmrotate - https://github.com/ShovalMishal/mmrotate/tree/dev-1.x

mmdet - https://github.com/ShovalMishal/mmdection/tree/dev-3.x

Then you should create an environment based on the requirements file here.
you should install them in the new environment according to their README.

There is a preprocess stage which aims to normalize the aerial images according to gsd values.
Here is a link to the final validation and train datasets you should use to run the program.


## Training and testing full pipeline
You need to update the relevant configurations in the config file, which is in this repo and supply it to the main script by running the following command:
```shell
python ./FullOODPipeline.py -c ./config.py 
``` 
Then, the anomaly detection runs and extracting objective bounding boxes according to dino vit results.
After, the classifier is fine-tuned for the OOD stage. 
Finally, you get the statistics for the OOD detection experiment.
