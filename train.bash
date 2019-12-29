#!/bin/bash

python frame2json.py /Users/marius/Documents/tinyAP/frame.2019-11-20.csv  \
	| python model.py \
	| python parse.py
