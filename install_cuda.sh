#!/bin/bash

pip uninstall torch torchvision -y

# Change the URL below with your desired CUDA version, check which one at https://pytorch.org/get-started/locally/
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
