#!/bin/bash

NAME=$1
PORT=$2

USER_NAME=$(basename $HOME)
echo "Run as user '$USER_NAME'"

HOST_PATH=$(readlink -f "$PWD/../../")
DOCKER_PATH="/root/$NAME"

cd $HOST_PATH

(docker container run \
    --rm \
    -dit \
    --gpus all \
    --shm-size=48G \
    --dns 217.10.39.4 --dns 8.8.8.8 \
    --privileged \
    --env="DISPLAY" \
    --workdir="/home/$USER" \
    --volume="/home/$USER:/home/$USER" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v $HOST_PATH:/$DOCKER_PATH \
    -v $HOME:/home/$USER_NAME \
    --expose $PORT \
    -p $PORT:$PORT \
    -h $NAME \
    --name $NAME \
    $NAME \
) || true

docker container exec -it -w $DOCKER_PATH $NAME bash

