# Docker instructions
1. Build the docker:
```shell
docker build --no-cache -t ood-final:run_ms_experiment -f docker/run_ms_experiment/Dockerfile .
docker tag ood-final:run_ms_experiment shovalmishal/ood-final:run_ms_experiment
docker push shovalmishal/ood-final:run_ms_experiment
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all --shm-size=24g -v /home/shoval/Documents/Repositories/:/home/shoval/Documents/Repositories/ --rm -it shovalmishal/ood-final:run_ms_experiment
```
On runai:
```shell
runai submit --name full-ood-pipeline -g 1.0 -i shovalmishal/ood-final:run_ms_experiment --pvc=storage:/storage --large-shm 
```

