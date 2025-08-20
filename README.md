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


Quick Start to run the repo

1. Generate dataset
   Run the simulator to generate LiDAR training and testing data:

   python lidar_sim_3d_dataset.py

   This will create train.csv and test.csv inside the project directory.
2. Preprocess Data
   Preprocess the generated datasets (scaling + encoding):

   python preprocess_lidar.py

   This will also save the scaler and encoders (scaler.pkl, relvel_encoder.pkl) for later use.

3. Check Label distribution (optional, if you want to know the number of individual cases generated in both test and train)
   Inspect how balanced the dataset is:

   python check_labels.py

4. Train Models
   Train and evaluate both the collision risk model and the maneuver recommendation model:

   python train_models.py

5. Export LiDAR scans to PCD
   If you want point cloud files for visualization

   python export_pcd.py

6. Generate visualizations (optional, to generate plots of the data used for training and testing)

   python figures.py
