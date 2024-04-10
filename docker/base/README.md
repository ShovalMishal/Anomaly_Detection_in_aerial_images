# Docker instructions
1. Build the docker:
```shell
docker build -t full-pipeline:base -f docker/base/Dockerfile .
docker tag full-pipeline:base shovalmishal/full-pipeline:base
docker push shovalmishal/full-pipeline:base
```
