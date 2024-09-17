# Docker instructions
1. Build the docker:
```shell
docker build --no-cache -t ood-final:run_ss_experiment -f docker/run_ss_experiment/Dockerfile .
docker tag ood-final:run_ss_experiment shovalmishal/ood-final:run_ss_experiment
docker push shovalmishal/ood-final:run_ss_experiment
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all --shm-size=24g -v /home/shoval/Documents/Repositories/:/home/shoval/Documents/Repositories/ --rm -it shovalmishal/ood-final:run_ss_experiment
```
On runai:
```shell
runai submit --name full-ood-pipeline-vit -g 1.0 -i shovalmishal/ood-final:train_regressor --pvc=storage:/storage --large-shm 
```

