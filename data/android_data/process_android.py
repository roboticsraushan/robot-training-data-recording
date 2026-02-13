import json
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import torch
import numpy as np
from scipy.integrate import cumulative_trapezoid
import open3d as o3d
import os
from PIL import Image

def load_data():
    with open('motion_data.json') as f:
        motion = json.load(f)
    with open('video_timestamps.json') as f:
        timestamps = json.load(f)['timestamps']
    cap = cv2.VideoCapture('video.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return motion, timestamps, cap, fps, total_frames

def initialize_models():
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    
    # Use new MediaPipe API
    base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    hands = vision.HandLandmarker.create_from_options(options)
    
    camera_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    return midas, midas_transforms, hands, camera_matrix

def process_imu(motion):
    data = motion['data']
    imu_timestamps = np.array([e[0] for e in data])
    gyro = np.array([e[1:4] for e in data])
    accel = np.array([e[4:7] for e in data])
    dt = np.diff(imu_timestamps, prepend=imu_timestamps[0])
    orientation = np.cumsum(gyro * dt[:, None], axis=0)
    velocity = np.array([cumulative_trapezoid(accel[:, i], imu_timestamps, initial=0) for i in range(3)]).T
    position = np.array([cumulative_trapezoid(velocity[:, i], imu_timestamps, initial=0) for i in range(3)]).T
    return imu_timestamps, orientation, position

def track_features(prev_frame, gray, prev_features, camera_matrix, current_pose):
    if prev_frame is not None:
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, gray, prev_features, None)
        good_old = prev_features[status == 1]
        good_new = next_pts[status == 1]
        if len(good_old) > 10:
            E, mask = cv2.findEssentialMat(good_new, good_old, camera_matrix, cv2.RANSAC, 0.999, 1.0)
            if E is not None:
                _, R, t, mask = cv2.recoverPose(E, good_new, good_old, camera_matrix)
                delta_pose = np.eye(4)
                delta_pose[:3, :3] = R
                delta_pose[:3, 3] = t.flatten()
                current_pose = current_pose @ delta_pose
    return current_pose

def detect_hands(hands, rgb):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = hands.detect(mp_image)
    hand_data = {'left': None, 'right': None}
    if results.hand_landmarks:
        for i, hand_landmarks in enumerate(results.hand_landmarks):
            label = results.handedness[i][0].display_name.lower() if results.handedness else ('left' if i % 2 == 0 else 'right')
            landmarks_2d = np.array([[lm.x, lm.y] for lm in hand_landmarks])
            confidence = results.handedness[i][0].score if results.handedness else 0.9
            hand_data[label] = {'landmarks_2d': landmarks_2d, 'confidence': confidence}
    return hand_data

def estimate_depth(midas, midas_transforms, rgb):
    # MiDaS expects numpy array (H, W, 3) in RGB format
    input_batch = midas_transforms(rgb)
    with torch.no_grad():
        depth = midas(input_batch).squeeze().cpu().numpy()
    return depth

def lift_hands_to_3d(hand_data, depth, camera_matrix, image_shape=None):
    """Lift 2D hand landmarks to 3D using depth map.
    - Convert MediaPipe normalized coords to *image* pixel coordinates (for correct overlay).
    - Sample the MiDaS depth map by mapping image pixels into depth-map coordinates.
    - Unproject using camera intrinsics (which expect image pixels).

    Args:
        image_shape: optional (height, width) of the original RGB frame. If omitted,
                     depth.shape will be used as a fallback (but that may cause misalignment).
    """
    # Depth map resolution
    depth_h, depth_w = depth.shape[:2]

    # Image resolution (use provided image size when available)
    if image_shape is not None:
        img_h, img_w = image_shape
    else:
        # fallback (not ideal) — use depth resolution
        img_h, img_w = depth_h, depth_w

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    for hand in ['left', 'right']:
        if hand_data[hand]:
            landmarks_3d = []
            landmarks_2d_pixels = []  # Store original pixel coordinates (image-space)

            for x_norm, y_norm in hand_data[hand]['landmarks_2d']:
                # --- convert normalized MediaPipe coords (0..1) to IMAGE pixel coords ---
                x_px_img = int(round(x_norm * img_w))
                y_px_img = int(round(y_norm * img_h))
                x_px_img = int(np.clip(x_px_img, 0, img_w - 1))
                y_px_img = int(np.clip(y_px_img, 0, img_h - 1))

                # store image-space pixel coords for accurate overlay drawing later
                landmarks_2d_pixels.append([x_px_img, y_px_img])

                # --- sample depth: map image pixels into depth-map coordinates (bilinear interpolation) ---
                x_depth_f = (x_px_img * (depth_w / img_w))
                y_depth_f = (y_px_img * (depth_h / img_h))

                # Bilinear sample from the (smaller) MiDaS depth map for smoother values
                x0 = int(np.floor(x_depth_f))
                y0 = int(np.floor(y_depth_f))
                x1 = min(x0 + 1, depth_w - 1)
                y1 = min(y0 + 1, depth_h - 1)
                dx = x_depth_f - x0
                dy = y_depth_f - y0

                v00 = float(depth[y0, x0])
                v10 = float(depth[y0, x1])
                v01 = float(depth[y1, x0])
                v11 = float(depth[y1, x1])
                depth_value = (v00 * (1 - dx) * (1 - dy)
                               + v10 * dx * (1 - dy)
                               + v01 * (1 - dx) * dy
                               + v11 * dx * dy)

                # Convert MiDaS inverse-depth-ish output to approximate metric z (empirical)
                z = 0.5 / (depth_value + 1e-6)

                # Unproject using IMAGE pixel coords (camera intrinsics are image-based)
                x_3d = (x_px_img - cx) * z / fx
                y_3d = (y_px_img - cy) * z / fy
                landmarks_3d.append([x_3d, y_3d, z])

            hand_data[hand]['landmarks_3d'] = np.array(landmarks_3d)
            hand_data[hand]['landmarks_2d_pixels'] = np.array(landmarks_2d_pixels)
    return hand_data

def process_frames(cap, timestamps, hands, midas, midas_transforms, camera_matrix, total_frames, imu_poses=None):
    """Process video frames with hand tracking, depth estimation, and pose tracking.
    
    Args:
        imu_poses: Optional IMU-based poses for fusion/fallback (currently unused but available)
    """
    frames = []
    frame_idx = 0
    current_pose = np.eye(4)
    prev_frame = None
    prev_features = None
    
    # Process every Nth frame for efficiency (sample rate)
    sample_rate = 1  # Process every frame for full-length video
    
    print(f"Processing {total_frames} frames (sampling every {sample_rate}th frame)...")
    
    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if we have timestamp for this frame
        if frame_idx >= len(timestamps):
            print(f"Warning: Frame {frame_idx} has no timestamp, stopping.")
            break
        
        # Sample frames for efficiency
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Visual odometry for camera pose estimation
        current_pose = track_features(prev_frame, gray, prev_features, camera_matrix, current_pose)
        prev_features = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
        prev_frame = gray
        
        # Hand detection and depth
        hand_data = detect_hands(hands, rgb)
        depth = estimate_depth(midas, midas_transforms, rgb)
        # Pass the original image size so 2D pixel coords are computed in IMAGE space
        hand_data = lift_hands_to_3d(hand_data, depth, camera_matrix, image_shape=rgb.shape[:2])
        
        t = timestamps[frame_idx]
        
        frames.append({
            'frame_id': frame_idx,
            'timestamp': t,
            'camera_pose_world': current_pose.tolist(),
            'hands': {
                'left': {
                    'joints': hand_data['left']['landmarks_3d'].tolist() if hand_data['left'] and 'landmarks_3d' in hand_data['left'] else [],
                    'joints_2d': hand_data['left']['landmarks_2d_pixels'].tolist() if hand_data['left'] and 'landmarks_2d_pixels' in hand_data['left'] else [],
                    'confidence': hand_data['left']['confidence'] if hand_data['left'] else 0
                },
                'right': {
                    'joints': hand_data['right']['landmarks_3d'].tolist() if hand_data['right'] and 'landmarks_3d' in hand_data['right'] else [],
                    'joints_2d': hand_data['right']['landmarks_2d_pixels'].tolist() if hand_data['right'] and 'landmarks_2d_pixels' in hand_data['right'] else [],
                    'confidence': hand_data['right']['confidence'] if hand_data['right'] else 0
                }
            }
        })
        
        if frame_idx % 30 == 0:
            print(f"  Processed frame {frame_idx}/{total_frames}")
        
        frame_idx += 1
    
    cap.release()
    print(f"Processing complete: {len(frames)} frames processed.")
    return frames

def save_unified_json(frames, fps):
    unified = {
        'sequence_metadata': {'device': 'android_estimated', 'fps': fps, 'depth_source': 'midas', 'format_version': 'v2_android'},
        'camera_intrinsics': {'fx': 500, 'fy': 500, 'cx': 320, 'cy': 240, 'width': 640, 'height': 480},
        'hand_joint_names': ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip", "index_dip", "index_tip", "middle_mcp", "middle_pip", "middle_dip", "middle_tip", "ring_mcp", "ring_pip", "ring_dip", "ring_tip", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_pip"],
        'frames': frames
    }
    os.makedirs('output', exist_ok=True)
    with open('output/unified_hand_tracking_android.json', 'w') as f:
        json.dump(unified, f)

def generate_visualization(frames):
    """Generate 3D visualization video for all processed frames.
    Visualizes:
    - Left hand joints (red)
    - Right hand joints (blue)
    - Camera/head position (green sphere)
    - Camera orientation (coordinate axes)
    """
    os.makedirs('output', exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    
    print(f"   - Rendering {len(frames)} frames to 3D visualization...")
    
    # Use sequential numbering for ffmpeg (0, 1, 2, 3...)
    for viz_idx, frame in enumerate(frames):
        geometries = []
        
        # Create point cloud for hands
        hand_points = []
        hand_colors = []
        
        # Left hand (red)
        if frame['hands']['left'] and len(frame['hands']['left']['joints']) > 0:
            for joint in frame['hands']['left']['joints']:
                hand_points.append(joint)
                hand_colors.append([1.0, 0.0, 0.0])  # Red
        
        # Right hand (blue)
        if frame['hands']['right'] and len(frame['hands']['right']['joints']) > 0:
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
        camera_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # Green
        camera_sphere.translate(camera_position)
        geometries.append(camera_sphere)
        
        # Camera orientation as coordinate frame (RGB = XYZ)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        coord_frame.transform(pose_matrix)
        geometries.append(coord_frame)
        
        # Add all geometries to visualizer
        for geom in geometries:
            vis.add_geometry(geom)
        
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer()
        img = (np.asarray(img) * 255).astype(np.uint8)
        # Use sequential index for ffmpeg, not original frame_id
        cv2.imwrite(f'/tmp/frame_{viz_idx:04d}.png', img)
        
        # Clear geometries for next frame
        for geom in geometries:
            vis.remove_geometry(geom)
        
        if viz_idx % 20 == 0:
            print(f"     Rendered {viz_idx}/{len(frames)} frames")
    
    vis.destroy_window()
    
    print(f"   - Encoding video...")
    # Use quiet mode for ffmpeg to reduce output
    # Match original video framerate (30 fps)
    os.system('ffmpeg -y -framerate 30 -i /tmp/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output/visualization_android.mp4 2>&1 | grep -E "(frame=|Duration)"')
    os.system('rm /tmp/frame_*.png')

def main():
    print("=" * 60)
    print("Android Hand Tracking Data Processing")
    print("=" * 60)
    
    print("\n1. Loading data...")
    motion, timestamps, cap, fps, total_frames = load_data()
    print(f"   - Video: {total_frames} frames at {fps} FPS")
    print(f"   - Timestamps: {len(timestamps)} entries")
    print(f"   - IMU samples: {motion['metadata']['sample_count']}")
    
    # Process all frames (no limit for production)
    max_frames = total_frames
    print(f"   - Processing all {max_frames} frames")
    
    print("\n2. Initializing models...")
    midas, midas_transforms, hands, camera_matrix = initialize_models()
    print("   - MiDaS depth estimator loaded")
    print("   - MediaPipe hand tracker loaded")
    
    print("\n3. Processing IMU data...")
    imu_timestamps, orientation, position = process_imu(motion)
    print(f"   - Computed poses for {len(imu_timestamps)} IMU samples")
    print("   - Note: IMU available for future fusion with visual odometry")
    
    print("\n4. Processing video frames...")
    frames = process_frames(cap, timestamps, hands, midas, midas_transforms, camera_matrix, max_frames)
    
    print("\n5. Saving unified JSON...")
    save_unified_json(frames, fps)
    print("   - Saved to: output/unified_hand_tracking_android.json")
    
    print("\n6. Generating visualization...")
    generate_visualization(frames)
    print("   - Saved to: output/visualization_android.mp4")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()