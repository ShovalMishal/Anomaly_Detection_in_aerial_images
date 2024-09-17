import json
import subprocess


class PrepareDataPyramid:
    def __init__(self, config):
        self.skip_stage = config.skip_stage
        self.use_gsd = config.use_gsd
        self.normalize_sizes = config.normalize_sizes
        self.config_template = config.config_template
        with open(self.config_template, 'r') as file:
            self.data = json.load(file)


    def prepare_data_pyramid(self):
        if self.use_gsd:
            for normalize_gsd in self.normalize_sizes:
                for data_type in ["train", "val", "test"]:
                    self.data["img_dirs"][0] = self.data["img_dirs"][0][:self.data["img_dirs"][0].find("DOTAV2/")+7]+data_type+"/images"
                    self.data["ann_dirs"][0] = self.data["ann_dirs"][0][:self.data["ann_dirs"][0].find("DOTAV2/")+7]+data_type+"/labelTxt"
                    self.data["metadata_dirs"][0] = self.data["metadata_dirs"][0][:self.data["metadata_dirs"][0].find("DOTAV2/")+7]+data_type+"/meta"
                    self.data["save_dir"] = self.data["save_dir"][:self.data["save_dir"].find("data/")+5] +f"multiscale_normalized_dataset_rotated/{data_type}"
                    self.data["target_gsd"] = normalize_gsd
                    self.data["normalize_accord_gsd"]=True
                    self.data["normalize_without_gsd"]=False
                    with open(self.config_template, 'w') as file:
                        json.dump(self.data, file, indent=4)

                    args = ['--base-json', self.config_template]
                    # Call script_a.py with the arguments
                    subprocess.run(['python', './mmrotate/tools/data/dota/split/img_split.py'] + args)

        else:
            for normalize_gsd in self.normalize_sizes:
                for data_type in ["train", "val", "test"]:
                    self.data["img_dirs"][0] = self.data["img_dirs"][0][:self.data["img_dirs"][0].find("DOTAV2/")+7]+data_type+"/images"
                    self.data["ann_dirs"][0] = self.data["ann_dirs"][0][:self.data["ann_dirs"][0].find("DOTAV2/")+7]+data_type+"/labelTxt"
                    self.data["metadata_dirs"][0] = self.data["metadata_dirs"][0][:self.data["metadata_dirs"][0].find("DOTAV2/")+7]+data_type+"/meta"
                    self.data["save_dir"] = self.data["save_dir"][:self.data["save_dir"].find("data/")+5] +f"multiscale_normalized_dataset_rotated_wo_gsd/{data_type}"
                    self.data["target_gsd"] = normalize_gsd
                    self.data["normalize_accord_gsd"] = False
                    self.data["normalize_without_gsd"] = True
                    with open(self.config_template, 'w') as file:
                        json.dump(self.data, file, indent=4)

                    args = ['--base-json', self.config_template]
                    # Call script_a.py with the arguments
                    subprocess.run(['python', './mmrotate/tools/data/dota/split/img_split.py'] + args)

    def get_original_data_path(self):
        return self.data["img_dirs"][0][:self.data["img_dirs"][0].find("DOTAV2/")+7]

    def get_lowest_gsd(self):
        return self.normalize_sizes[0]