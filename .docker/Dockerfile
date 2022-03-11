FROM tensorflow/tensorflow:latest-gpu

RUN pip install -r requirements.txt

RUN apt-get -y update
RUN apt-get -y install git

RUN useradd -s /bin/bash -m vscode \
 && groupadd docker \
 && usermod -aG docker vscode

USER vscode

