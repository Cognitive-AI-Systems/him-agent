#!/bin/bash
set -euo pipefail

script_dir=$(dirname "$0")


# Note that built images are pushed to personal _public_ DockerHub repo
#   pkuderov/xxx

# Template is based on the https://pythonspeed.com/articles/faster-multi-stage-builds/ post

# --------- Uncomment if you use old Docker build builder --------
# # Pull the latest version of the image, in order to
# # populate the build cache. This isn't necessary if 
# # you're using BuildKit.
# docker pull pkuderov/htm-core-fork:latest || true
# docker pull pkuderov/hima:latest || true

# --------- Build htm-core-fork ------------:
docker build \
    --target htm-core-fork \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --cache-from=pkuderov/htm-core-fork:latest \
    --tag pkuderov/htm-core-fork:latest \
    -f $script_dir/Dockerfile \
    $PWD

# ------------- Build hima -----------------:
docker build \
    --target hima \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --cache-from=pkuderov/htm-core-fork:latest \
    --cache-from=pkuderov/hima:latest \
    --tag pkuderov/hima:latest \
    -f $script_dir/Dockerfile \
    $PWD

# Push the new versions:
docker push pkuderov/htm-core-fork:latest
docker push pkuderov/hima:latest