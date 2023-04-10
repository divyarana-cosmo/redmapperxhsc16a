#!/bin/bash
module load anaconda2 mpich-3.3.1 gcc-8.2.0 gsl-2.6
python count_pairs.py --output random_pairs.dat --srcrand True --mirror True



