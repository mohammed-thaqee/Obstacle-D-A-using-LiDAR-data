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

Project Structure

.
├── .gitignore              # Git ignore file  
├── README.md               # Project documentation  
├── check_labels.py         # Utility to check dataset label distribution  
├── export_pcd.py           # Export LiDAR scans into PCD format  
├── figures.py              # Script to generate plots and figures  
├── lidar_sim_3d_dataset.py # LiDAR simulation and dataset generation  
├── preprocess_lidar.py     # Data preprocessing (scaling, encoding)  
├── train_models.py         # Training and evaluation of ML models (Risk + Maneuver)  
├── test.csv                # Sample test dataset  
├── scan_0000.pcd           # Example LiDAR point cloud file  
├── scan_0001.pcd           # Example LiDAR point cloud file  
├── scan_0002.pcd  
├── scan_0003.pcd  
├── scan_0004.pcd  
├── scan_0005.pcd  

Requirements:

This project is implemented in Python 3.8+ and uses standard machine learning and data processing libraries.

Install the dependencies using:

pip install -r requirements.txt


Usage:

Follow these steps to generate the dataset, preprocess it, train the models, and evaluate performance.

1. Generate LiDAR Simulation Data
   This step creates synthetic 3D LiDAR scans with obstacles and corresponding maneuver labels.

   python lidar_sim_3d_dataset.py

   Generates train.csv and test.csv inside the project folder.
   Also produces .pcd files (scan_0000.pcd, scan_0001.pcd, …) that represent LiDAR point clouds.

2. Check Label Distribution
   To inspect the distribution of maneuvers (maintain, stop, turn_left, turn_right) in the generated dataset:

   python check_labels.py

3. Preprocess Data
   Convert raw LiDAR simulation data into numerical features for training and testing.

   python preprocess_lidar.py

   Outputs processed features.
   Saves preprocessing artifacts like scaler and encoders (scaler.pkl, relvel_encoder.pkl).

4. Train & Evaluate Models
   Train machine learning models (Random Forest) for:

   Collision Risk Prediction (binary: risk/no risk)
   Maneuver Recommendation (multi-class: maintain, stop, turn_left, turn_right)

   python train_models.py

   This prints:

   Classification Reports (precision, recall, f1-score)
   Confusion Matrices for collision risk and maneuver prediction.

5. Export Point Clouds (Optional)
   Convert LiDAR data to .pcd format for visualization in external tools (e.g., CloudCompare, Meshlab).

   python export_pcd.py

6. Generate visualizations (optional, to generate plots of the data used for training and testing)

   python figures.py

Future Work:

This project serves as a foundational step toward autonomous navigation using LiDAR data. Several enhancements can be pursued in the future:

Improved Data Generation

Balance maneuver classes (stop, turn_left, turn_right) further.
Introduce more diverse obstacle shapes, speeds, and trajectories.

Advanced Models

Extend beyond Random Forests to Deep Learning approaches (e.g., CNNs, RNNs, Transformers) for spatiotemporal LiDAR analysis.
Explore reinforcement learning for decision-making.

Realistic Simulation

Integrate with Gazebo / CARLA simulators to model realistic robot or autonomous car movement.
Simulate both stationary and moving robot scenarios.

Real-world Deployment

Connect to actual LiDAR sensor hardware.
Deploy trained models on embedded devices for real-time obstacle detection and avoidance.

Performance Improvements

Use class weighting or SMOTE oversampling to address imbalance.
Experiment with ensemble learning and hyperparameter tuning for higher F1-scores.


I hope to continuosly add on to the work I have done so far. I am also eager on bringing this to a fully functional prototype. If you'd like to contribute or collaborate it would be great. You can find my LinkedIn profile here:
