#!/bin/bash

purge
for i in {0..9}
do
	python analysis_frankenstein.py $i >> frank_output.txt
  purge
done
