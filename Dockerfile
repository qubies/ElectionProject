FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN pip3 install transformers pandas sklearn 
RUN pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html
COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r requirements.txt

