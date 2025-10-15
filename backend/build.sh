#!/usr/bin/env bash
# exit on error
set -o errexit

# Install the python dependencies
pip install -r requirements.txt

# Download the spaCy model
python -m spacy download en_core_web_sm