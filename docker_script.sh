#!/bin/bash

conda install python=3.12

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update
apt-get -y install cudnn9-cuda-12


echo "Step.1 - installing fastapi uvicorn onnxruntime-genai-cuda"
pip install fastapi uvicorn onnxruntime-genai-cuda

echo "step.1b - installing baselina model requirement"
pip install torch==2.3.1
pip install flash_attn accelerate==0.31.0 transformers==4.43.0

echo "Step.2 - azure-cli"
pip install --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge

echo "Step.3 - azure-ai"
pip install azure-ai-evaluation --upgrade

echo "Step.4 - promptflow-azure"
pip install promptflow-azure --upgrade

az login --identity
echo "print out az account"
az account show

cd /ort_src
python ./docker_main.py