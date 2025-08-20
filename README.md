# Obstacle-D-A-using-LiDAR-data
Machine Learning Model for Obstacle Detection and Avoidance using LiDAR Data


Introduction:

This project implements a simulation-based framework for obstacle detection and avoidance using LiDAR data.
A virtual self-driving robot is placed in synthetic 3D driving scenarios with randomly generated obstacles.
At each timestep, the robot receives a LiDAR scan, processes the data into features, and predicts:

Collision Risk – Whether the current situation poses a threat of collision.

Recommended Maneuver – The optimal driving action (maintain, stop, turn left, or turn right).

We trained and evaluated machine learning models (Logistic Regression and Random Forest) on the generated data.
The system achieves ~91% accuracy in collision risk prediction and ~76% accuracy in maneuver recommendation, showing the potential of combining simulation + ML for autonomous navigation tasks.

