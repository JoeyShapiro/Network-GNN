#!/bin/bash
# you need conda and python (duh)

# setup env
conda create -n its472 python=3.9
conda activate its472
conda install pytorch torchvision torchaudio cudatoolkit11.3 -c pytorch
conda install pandas matplotlib
conda install -c dglteam dgl-cuda11.3
conda install -c anaconda pydot