#!/bin/bash
kill $(ps aux | grep '/home/s144234/.conda/envs/py3_env/bin/tensorboard' | awk '{print $2}')
