FROM tensorflow/tensorflow:latest-gpu

COPY requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt

RUN apt-get -y update
RUN apt-get -y install git

RUN useradd -s /bin/bash -m vscode \
 && groupadd docker \
 && usermod -aG docker vscode

USER vscode

