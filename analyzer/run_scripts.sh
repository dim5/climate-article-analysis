#!/bin/bash

cd src/pre-process
python replacer.py
python preprocess.py
python sentencizer.py


cd ../process
python process.py
python clusterer.py
python graph-create.py