# drone_to_drone_winddetection

This set of programs analyzes data collected from one drone flying underneath another.
The goal of this project was to be able detect when the drone is over underneath the other drone
using data mining techniques. This is the dat mining code.

Drone1: This is where the data is stored (in .csv files)
Transformers: This is for feature generation
Utils: Various code for noise reduction, data output visuals, and grid search
One_second_slice.py: this is the code that trains on the data and tests on it and reports a score
(for each second of flight time)

Please note that is was a collabrative research project for the Fordham University Robotics and Computer Vision Lab
and not all the was written by me
