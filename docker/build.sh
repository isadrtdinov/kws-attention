#!/bin/bash

NAME=$1

docker image build -t $NAME docker/

