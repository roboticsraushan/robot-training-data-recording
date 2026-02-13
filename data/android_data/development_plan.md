# Android Data Processing Plan

## Overview
This plan outlines generating output similar to the iPhone-processed data (unified JSON and 3D visualization) using Android data. Key constraints: No depth sensors, hand landmarks, or AR session data. We'll approximate with MediaPipe for hands, monocular depth estimation for 3D, and IMU for head tracking.

## Available Data
- `motion_data.json`: IMU (accel, gyro) for head/camera tracking.
- `video.mov`: RGB video for hand tracking and depth estimation.
- `video_timestamps.json`: Frame timestamps for synchronization.
- `metadata.json`: Basic info (device, fps).

## Missing Data (Accepted Limitations)
- `hands.json`: Approximated via MediaPipe.
- `depth_metadata.json` & `depth_maps/`: Approximated via monocular depth (e.g., MiDaS).
- `ar_session.json`: Approximated via IMU integration for poses.

## Processing Flowchart
```
[Start]
    |
    v
[Load Android Data]
    - motion_data.json (IMU timestamps, gyro, accel)
    - video_timestamps.json (frame timestamps)
    - video.mov (RGB frames)
    |
    v
[Initialize Models]
    - Load MiDaS (depth estimation)
    - Setup MediaPipe Hands
    |
    v
[IMU Integration for Head Tracking]
    - Compute orientation (cumsum gyro * dt)
    - Compute velocity/position (integrate accel)
    - Fuse with visual odometry for accurate camera poses
    |
    v
[Visual SLAM/Odometry for Head Tracking]
    - Track features across frames (e.g., ORB features, optical flow)
    - Estimate camera motion (rotation/translation) per frame
    - Use as reference to correct IMU drift; head as origin
    |
    v
[Frame-by-Frame Processing Loop]
    For each video frame:
        - Read frame -> RGB
        - Hand Tracking: MediaPipe -> 2D landmarks (left/right)
        - Depth Estimation: MiDaS -> depth map
        - Lift 2D to 3D: Sample depth at landmarks, scale to meters
        - Pose Interpolation: Match IMU pose to frame timestamp
        - Store: frame_id, timestamp, camera_pose_world, hands (joints + confidence)
    |
    v
[Save Unified JSON]
    - Structure: metadata, intrinsics, joint names, frames
    - Output: ../output/unified_hand_tracking_android.json
    |
    v
[Generate 3D Visualization Video]
    - For sample frames: Create point cloud from hand joints
    - Render with Open3D, save images
    - Compile to MP4 with ffmpeg
    - Output: ../output/visualization_android.mp4
    |
    v
[End]
```

## Key Steps
1. **Head Tracking**: Use visual SLAM/odometry on RGB video for camera poses (reference-based), fused with IMU for smoothing. Head (camera) tracked as origin.
2. **Hand Tracking**: Use MediaPipe on video for 2D landmarks relative to head.
3. **Depth Estimation**: Apply MiDaS to RGB frames for relative depth; lift 2D hands to 3D.
4. **Fusion**: Transform hands to world coords using head poses.
5. **Output**: Unified JSON (android-adapted) + MP4 visualization.

## Depth Estimation Method
- Use MiDaS (monocular ML model) for depth maps from RGB.
- Sample depth at hand joints; calibrate for scale.
- Alternatives: Depth Anything or stereo if dual-camera.

## Tools/Libraries
- Python: OpenCV, MediaPipe, PyTorch (for MiDaS), NumPy/SciPy (IMU), Open3D (viz).

## Accuracy Notes
- Lower than iPhone (no LiDAR); depth errors ~10-20%.
- Test on sample frames; refine with calibration.

## Next Steps
- Implement script for processing.
- Validate on sample data.
- Iterate based on results.