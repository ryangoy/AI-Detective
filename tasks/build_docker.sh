#!/bin/bash

# Compile Pipenv env packages
pipenv lock --requirements --keep-outdated > web/requirements.txt

# Hack to change tensorflow-gpu to tensorflow (cpu)
sed -i 's/-gpu//g' web/requirements.txt

# Call docker
docker build -t lie_detector_api -f web/Dockerfile .
