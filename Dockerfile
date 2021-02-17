FROM tensorflow/tensorflow:2.3.2-gpu

ENV INSTALL_DIR /orcanet
ADD . $INSTALL_DIR
RUN apt-get update
RUN apt-get -y install python3-tk python3-venv git
RUN cd $INSTALL_DIR && make install
RUN pip install ipython
WORKDIR /orcanet
ENTRYPOINT /bin/bash
