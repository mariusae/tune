#!/bin/bash

for f in *_test.py
do
	echo $f
	python $f
done
