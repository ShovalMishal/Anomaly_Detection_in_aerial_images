# Docker instructions
1. Build the docker:
```shell
docker build --no-cache -t ood-final:base -f docker/base/Dockerfile .
docker tag ood-final:base shovalmishal/ood-final:base
docker push shovalmishal/ood-final:base
```
