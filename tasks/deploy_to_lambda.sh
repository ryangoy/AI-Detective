#!/bin/bash

rm -r web/.serverless
export AWS_PROFILE="default" && export AWS_REGION=eu-west-1
pipenv lock --requirements --keep-outdated > web/requirements.txt
sed -i 's/tensorflow-gpu/tensorflow/' web/requirements.txt
cd web || exit 1
npm install
pipenv run sls deploy -v
