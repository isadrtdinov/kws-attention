#!/bin/bash

wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz -O speech_commands_v0.01.tar.gz
mkdir -p speech_commands && tar -C speech_commands -xzf speech_commands_v0.01.tar.gz
rm speech_commands_v0.01.tar.gz
rm -rf speech_commands/_background_noise_

