#!/bin/sh
echo Running lightSense script ...
python3 -u /home/pi/lightSense.py > /home/pi/lightSense_log.log 2&>1 &