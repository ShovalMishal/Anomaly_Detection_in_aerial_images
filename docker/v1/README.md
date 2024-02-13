# Docker instructions
1. Build the docker:
```shell
docker build -t full-pipeline:v1 -f docker/v1/Dockerfile .
docker tag full-pipeline:v1 shovalmishal/full-pipeline:v1
docker push shovalmishal/full-pipeline:v1
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/workspace/ -e EXPORT_SCRIPT=/workspace/scripts/what_to_say.sh  -e RUN_SCRIPT=/workspace/scripts/say_hello.sh  -it shovalmishal/ad-stage1:v1


```
On runai:
```shell
runai submit --name amir-autorep-pretrain -g 1.0 -i shovalmishal/ad-stage1:v1 -e EXPORT_SCRIPT=/storage/jevnisek/autorep/scripts/say_hello.sh -e RUN_SCRIPT=/storage/jevnisek/autorep/scripts/scripts_baseline.sh --pvc=storage:/storage --large-shm
runai submit --name amir-autorep-end2end -g 1.0 -i shovalmishal/ad-stage1:v1 -e EXPORT_SCRIPT=/storage/jevnisek/autorep/scripts/say_hello.sh -e RUN_SCRIPT=/storage/jevnisek/autorep/scripts/run_resnet18_cifar100_end_to_end.sh --pvc=storage:/storage --large-shm
```

