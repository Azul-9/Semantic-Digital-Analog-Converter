# Semantic-Digital-Analog-Converter

It's the code repository of the paper "sDAC---Semantic Digital Analog Converter for Semantic Communications". It contains the sDAC module along with a simple implementation example mentioned in the paper. This repo is organized as the follows:

* adapter.py: It contains the code of the sDAC, including an end-to-end forward() and split encode(), decode()
* bsc_channel.py: It contains the implementation of discrete BSC
* config.py: Some configs for the system
* datasets.py: A simple Vimeo dataset for training. Feel free to reimplement it according to your demands
* logger.py: Utils for logging
* network_adapter.py: A simple semantic image transmission framework which enabled by sDAC for performance validation
* subnets: Network parts of the network_adapter
* test_adapter.py: A simple script for performance validation
* train_adapter.py: A script for training based on DDP. Two GPUs are utilized in this script. It's recommended to modify this script according to hardware and other demands.

