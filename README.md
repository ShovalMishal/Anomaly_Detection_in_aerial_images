# Is there a needle in the Haystack?
We introduce a simple approach to identify novel class objects within aerial images, without specify these novel classes
ahead of time. In our setting, we are equipped with a detector capable of detecting a closed set of objects 
(e.g., vehicles, planes) but wish to determine if other, unspecified, object classes, that are of interest (say, ships),
appear in the images as well. we propose a funnel approach that gradually reduces the number of patches of interest from 
tens of millions to a short list of few tens of thousands. The patches in the short list are ranked automatically and 
shown to a human operator. We therefore measure performance by ``Time-To-$1^{st}$'' (TT-1), i.e. the time it takes a 
human to find the first instance of interesting new classes in aerial images, and show we are capable of producing such 
a sample within the first few patches.

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
