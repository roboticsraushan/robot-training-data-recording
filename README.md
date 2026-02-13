# robot-training-data-recording

## Data Sections

### 1. Raw Input
Data recorded from an iPhone for egocentric capture. Includes depth data from LiDAR, AR session data, hands.json detected via ARNet, motion data (IMU and gyro), and video.mov (RGB video).

### 2. Output
Processed data from iPhone inputs, including depth_map-overlay with depth and hand points, and visualization_complete_v2.mp4 (3D visualization with head and hand tracking). The unified_hand_tracking_complete_v2.json contains the processed hand tracking data.

### 3. Android Data
Data generated from a cheap Android mobile, copied from input data. Lacks depth data and high frame rate IMU.

## Goal
Generate output similar to the output folder using only Android data.