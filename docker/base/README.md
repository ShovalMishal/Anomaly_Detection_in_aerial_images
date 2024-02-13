# Docker instructions
1. Build the docker:
```shell
docker build -t full-pipeline:base -f docker/base/Dockerfile .
docker tag full-pipeline:base shovalmishal/full-pipeline:base
docker push shovalmishal/full-pipeline:base
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/what_to_say.sh  -e RUN_SCRIPT=/local_code/scripts/say_hello.sh  -it shovalmishal/ad-stage1:v3
docker run -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/what_to_say.sh  -e RUN_SCRIPT=/local_code/scripts/say_hello.sh  -it shovalmishal/ad-stage1:v3
docker run -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/what_to_say.sh  -e RUN_SCRIPT=/local_code/scripts/say_hello.sh  -it shovalmishal/ad-stage1:v3 /bin/bash 

```
On runai:
```shell
runai submit --name amir-autorep-pretrain -g 1.0 -i shovalmishal/ad-stage1:v3 -e EXPORT_SCRIPT=/storage/jevnisek/autorep/scripts/say_hello.sh -e RUN_SCRIPT=/storage/jevnisek/autorep/scripts/scripts_baseline.sh --pvc=storage:/storage --large-shm
runai submit --name amir-autorep-end2end -g 1.0 -i shovalmishal/ad-stage1:v3 -e EXPORT_SCRIPT=/storage/jevnisek/autorep/scripts/say_hello.sh -e RUN_SCRIPT=/storage/jevnisek/autorep/scripts/run_resnet18_cifar100_end_to_end.sh --pvc=storage:/storage --large-shm
```

