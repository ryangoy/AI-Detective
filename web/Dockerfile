  # Start from the latest Long Term Support (LTS) Ubuntu version
FROM ubuntu:18.04
USER root
# Install pipenv
RUN apt-get update && apt-get install python3-pip -y && pip3 install pipenv
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev
# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Copy only the relevant directories to the working diretory
COPY lie_detector/ ./lie_detector
COPY web/ ./web

# Install Python dependencies
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN set -ex && pip3 install -r web/requirements.txt

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo
CMD python3 /repo/web/server/server.py
