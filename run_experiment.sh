#!/bin/bash

chmod +x /mmrotate/tools/data/dota/split/img_split.py
# first prepare the data
python /mmrotate/tools/data/dota/split/img_split.py --base-json "./configs/experiment_2/ss_train.json"
python /mmrotate/tools/data/dota/split/img_split.py --base-json "./configs/experiment_2/ss_test.json"

# then create validation dataset from train dataset
python arrange_data.py --config "./configs/experiment_2/ss_train.json"

# running full pipeline
python ./FullOODPipeline.py -c "./configs/experiment_2/config.py"