FROM ubuntu
MAINTAINER cmbasnett@gmail.com
RUN apt-get update
RUN apt -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt -y install python3.9 python3-pip python3-virtualenv

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ADD . ./evepacker-api/
WORKDIR ./evepacker-api
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT python ./app.py
