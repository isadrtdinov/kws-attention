# Attention-based Model for Keywords Spotting

Paper: <https://arxiv.org/abs/1803.10916>

Data: <http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz>

## Docker

Build container:

`./docker/build.sh <container>`

Run container:

`./docker/run.sh <container> <port>`

Stop container:

`./docker/stop.sh <container>`

## Model utilization

Init project module:

`./scripts/init_module.sh`

Download training data:

`./scripts/download_data.sh`

Download model checkpoint:

`./scripts/download_model.sh`

Start training process:

`./scripts/train_model.sh`

Model inference, this one is configured to process `example.wav` file. Output probabilities are plotted in `example_probs.jpg` file:

`./scripts/test_model.sh`

## Results

Here is an example of model probs for `example.wav`. This audio contains the following spoken text: "Hello, dear Sheila! This message was recorded to you by Marvin himself".

<img src="/example_probs.jpg" alt="drawing" width="1000"/>
