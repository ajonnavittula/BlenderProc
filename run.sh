#!/bin/bash

for i in {2501..5000}
do
   blenderproc run main-box-physics.py --output-dir output --id ${i} --data-path /media/ws1/Data3/datasets/sps_synthetic
done