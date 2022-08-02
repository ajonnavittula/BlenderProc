#!/bin/bash

for i in {2501..5000}
do
   blenderproc run main-box-physics.py --output-dir output --id ${i}
   # blenderproc extract hdf5 output/0.hdf5
   # mv 0_colors.png ${i}_colors.png
   # mv 0_depth.png ${i}_depth.png
   # mv 0_instance_segmaps.png ${i}_instance_segmaps.png
done