#!/bin/bash
# you need conda and python (duh)

# setup env
conda create -n its472v4 python=3.8
conda activate its472v5
# conda install pytorch torchvision torchaudio cudatoolkit11.3 -c pytorch
# conda install pandas matplotlib
# conda install -c anaconda pydot
conda install -c conda-forge tensorflow
# # because on an m1 i could not do this part with conda :P
git clone https://github.com/stellargraph/stellargraph.git
conda activate its472v4
cd stellargraph
# if on m1, change line 27 from `f"{tensorflow}>=2.1.0",` to `f"{tensorflow}-macOS>=2.1.0",`
pip install .
conda install -c anaconda ipython
pip install numpy==1.19.2
# install tensor another way, mac m1 protlem again
# download .whl form `https://drive.google.com/drive/folders/1oSipZLnoeQB0Awz8U68KYeCPsULy_dQ7`
pip install ~/Downloads/tensorflow-2.4.1-py3-none-any.whl
conda install -c dglteam dgl
pip install torch