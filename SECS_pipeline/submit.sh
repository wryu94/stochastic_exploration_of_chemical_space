#!/bin/bash
nohup caffeinate -dims python SECS_pipeline.py > SECS_pipeline.out 2>&1 &
ps aux | grep SECS_pipeline.py.py
