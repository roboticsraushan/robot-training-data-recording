#!/usr/bin/env python3
"""
Quickly regenerate visualization from existing unified JSON.
This avoids reprocessing video frames.
"""

import json
import numpy as np
import open3d as o3d
import cv2
import os

def generate_visualization(json_path):
    """Generate 3D visualization video from unified JSON.
    Visualizes:
    - Left hand joints (red)
    - Right hand joints (blue)  
    - Camera/head position (green sphere)
    - Camera orientation (RGB coordinate axes = XYZ)
    """
    
    print("Loading JSON data...")
    with open(json_path) as f:
        data = json.load(f)
    
    frames = data['frames']
    print(f"Found {len(frames)} frames to visualize")
    
    os.makedirs('output', exist_ok=True)
    
    print("Creating 3D visualizations...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    
    rendered_count = 0
    for viz_idx, frame in enumerate(frames):
        geometries = []
        
        # Hand points with colors
        hand_points = []
        hand_colors = []
        
        # Add left hand (red)
        if frame['hands']['left']['joints']:
            for joint in frame['hands']['left']['joints']:
                hand_points.append(joint)
                hand_colors.append([1.0, 0.0, 0.0])  # Red
        
        # Add right hand (blue)
        if frame['hands']['right']['joints']:
            for joint in frame['hands']['right']['joints']:
                hand_points.append(joint)
                hand_colors.append([0.0, 0.0, 1.0])  # Blue
        
        if hand_points:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(hand_points)
            pcd.colors = o3d.utility.Vector3dVector(hand_colors)
            geometries.append(pcd)
        
        # Add camera/head position and orientation
        pose_matrix = np.array(frame['camera_pose_world'])
        camera_position = pose_matrix[:3, 3]
        
        # Camera position as green sphere
        camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        camera_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # Green for head/camera
        camera_sphere.translate(camera_position)
        geometries.append(camera_sphere)
        
        # Camera orientation as coordinate frame
        # Red=X, Green=Y, Blue=Z axes showing camera direction
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        coord_frame.transform(pose_matrix)
        geometries.append(coord_frame)
        
        # Add all geometries
        for geom in geometries:
            vis.add_geometry(geom)
        
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer()
        img = (np.asarray(img) * 255).astype(np.uint8)
        cv2.imwrite(f'/tmp/frame_{viz_idx:04d}.png', img)
        
        # Remove geometries for next frame
        for geom in geometries:
            vis.remove_geometry(geom)
        
        rendered_count += 1
        
        if viz_idx % 20 == 0:
            print(f"  Rendered {viz_idx}/{len(frames)} frames")
    
    vis.destroy_window()
    print(f"Rendered {rendered_count} frames total")
    
    print("Encoding video with ffmpeg...")
    result = os.system('ffmpeg -y -framerate 30 -i /tmp/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output/visualization_android.mp4 2>&1 | tail -5')
    
    if result == 0:
        print("✓ Video created successfully")
    
    print("Cleaning up temporary files...")
    os.system('rm /tmp/frame_*.png')
    
    print(f"✓ Done! Saved to: output/visualization_android.mp4")
    
    # Show file info
    os.system('ls -lh output/visualization_android.mp4')
    os.system('ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 output/visualization_android.mp4 2>/dev/null | xargs -I {} echo "Duration: {} seconds"')

if __name__ == "__main__":
    generate_visualization('output/unified_hand_tracking_android.json')
