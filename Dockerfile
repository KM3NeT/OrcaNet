FROM python:3.6.6

ENV INSTALL_DIR /orcanet
ADD . $INSTALL_DIR
RUN cd $INSTALL_DIR && make install
WORKDIR /orcanet/examples
ENTRYPOINT /bin/bash
