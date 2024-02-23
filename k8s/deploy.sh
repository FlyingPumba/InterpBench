#!/bin/bash

# build and push docker image
cd ..
docker build . -t iarcuschin/circuits-benchmark
docker push iarcuschin/circuits-benchmark
cd k8s

# delete all kubernete jobs
kubectl delete jobs --all

# create new job
kubectl create -f devbox.yaml