#!/bin/bash
python sm_run.py 150 --separate_dorms --separate_quarantine -t 14 -c 2 -l 75 -p 20 -k 0.99 -f 'Unit Test New LC Change' & 
sleep 3s
python sm_run.py 157 --separate_dorms --separate_quarantine -t 14 -c 2 -l 75 -p 20 -k 0.99 -f 'Unit Test New LC Change' & 
sleep 3s
python sm_run.py 164 --separate_dorms --separate_quarantine -t 14 -c 2 -l 75 -p 20 -k 0.99 -f 'Unit Test New LC Change' &
