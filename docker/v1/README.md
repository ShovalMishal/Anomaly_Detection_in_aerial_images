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
docker run --gpus all -v $(pwd):/workspace/ -it shovalmishal/ad-stage1:v1


```
On runai:
```shell
runai submit --name full-ood-pipeline-vit -g 1.0 -i shovalmishal/full-pipeline:v1 --pvc=storage:/storage --large-shm 
```

