FROM tensorflow/tensorflow:2.3.2-gpu

ENV INSTALL_DIR /orcanet
ADD . $INSTALL_DIR
RUN cd $INSTALL_DIR && make install
WORKDIR /orcanet
ENTRYPOINT /bin/bash
