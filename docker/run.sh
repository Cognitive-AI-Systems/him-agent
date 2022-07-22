#!/bin/bash
docker run \
    --platform linux/amd64 \
    --name hima \
    -v $PWD:/app/hima \
    -it \
    pkuderov/hima \
    zsh
