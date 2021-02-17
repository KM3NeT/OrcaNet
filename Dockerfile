FROM tensorflow/tensorflow:2.3.2-gpu

RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN apt-get -y install python3-tk python3-venv git
ENV INSTALL_DIR /orcanet
ADD . $INSTALL_DIR
RUN cd $INSTALL_DIR && make install
RUN pip install ipython
WORKDIR /orcanet
ENTRYPOINT /bin/bash
