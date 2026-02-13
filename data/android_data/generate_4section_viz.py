#!/usr/bin/env python3
"""
Generate comprehensive 4-section visualization:
1. Original video
2. MediaPipe hand overlay with connections
3. Depth values overlaid on hands
4. 3D matplotlib visualization of hand tracking
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import os

# Head-stability defaults (tunable)
HEAD_STABILITY_WINDOW = 3       # frames
HEAD_ROT_THRESH_DEG = 5.0       # degrees (lenient default)

# MediaPipe hand connections (indices for 21 landmarks)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17)
]

def load_data():
    """Load JSON data and video."""
    print("Loading JSON data...")
    with open('output/unified_hand_tracking_android.json') as f:
        data = json.load(f)
    
    frames_data = data['frames']
    camera_intrinsics = data['camera_intrinsics']
    print(f"Loaded {len(frames_data)} frames")
    print(f"Camera intrinsics: fx={camera_intrinsics['fx']}, fy={camera_intrinsics['fy']}, cx={camera_intrinsics['cx']}, cy={camera_intrinsics['cy']}")
    
    print("Loading video...")
    cap = cv2.VideoCapture('video.mp4')
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps} fps")
    
    return frames_data, cap, fps, width, height, camera_intrinsics

def draw_hand_landmarks(img, hand_data, color, camera_intrinsics, draw_depth=False):
    """Draw hand landmarks and connections on image."""
    if not hand_data['joints'] or hand_data['confidence'] == 0:
        return img
    
    landmarks_3d = hand_data['joints']
    
    # Use stored 2D pixel coordinates directly (most accurate)
    if 'joints_2d' in hand_data and hand_data['joints_2d']:
        landmarks_2d = [(int(x), int(y)) for x, y in hand_data['joints_2d']]
    else:
        # Fallback: project 3D to 2D (less accurate due to depth estimation errors)
        landmarks_2d = []
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
        
        for joint_3d in landmarks_3d:
            x_3d, y_3d, z_3d = joint_3d
            if z_3d > 0:
                x_px = int(x_3d * fx / z_3d + cx)
                y_px = int(y_3d * fy / z_3d + cy)
            else:
                x_px, y_px = int(cx), int(cy)
            
            x_px = max(0, min(img.shape[1] - 1, x_px))
            y_px = max(0, min(img.shape[0] - 1, y_px))
            landmarks_2d.append((x_px, y_px))
    
    # Draw connections
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks_2d) and end_idx < len(landmarks_2d):
            pt1 = landmarks_2d[start_idx]
            pt2 = landmarks_2d[end_idx]
            cv2.line(img, pt1, pt2, color, 2)
    
    # Draw landmarks
    for idx, (x, y) in enumerate(landmarks_2d):
        cv2.circle(img, (x, y), 4, color, -1)
        cv2.circle(img, (x, y), 5, (255, 255, 255), 1)
        
        # Draw depth values if requested
        if draw_depth and idx < len(landmarks_3d):
            depth = landmarks_3d[idx][2]  # z-coordinate
            depth_text = f"{depth*1000:.0f}mm"
            cv2.putText(img, depth_text, (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return img

def create_section1(frame):
    """Section 1: Original video."""
    return frame.copy()

def create_section2(frame, frame_data, camera_intrinsics):
    """Section 2: MediaPipe overlay with hand connections."""
    overlay = frame.copy()
    
    # Draw left hand (red)
    if frame_data['hands']['left']['joints']:
        overlay = draw_hand_landmarks(overlay, frame_data['hands']['left'], 
                                      (0, 0, 255), camera_intrinsics, draw_depth=False)
    
    # Draw right hand (blue)
    if frame_data['hands']['right']['joints']:
        overlay = draw_hand_landmarks(overlay, frame_data['hands']['right'], 
                                      (255, 0, 0), camera_intrinsics, draw_depth=False)
    
    # Add text overlay
    cv2.putText(overlay, "MediaPipe Hand Tracking", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show confidence scores
    if frame_data['hands']['left']['confidence'] > 0:
        conf_text = f"L: {frame_data['hands']['left']['confidence']:.2f}"
        cv2.putText(overlay, conf_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    if frame_data['hands']['right']['confidence'] > 0:
        conf_text = f"R: {frame_data['hands']['right']['confidence']:.2f}"
        cv2.putText(overlay, conf_text, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return overlay

def create_section3(frame, frame_data, camera_intrinsics):
    """Section 3: same as MediaPipe overlay (section 2) but show *only* wrist depth
    for each detected hand (displayed on the wrist joint). No other depth text or
    overlays are added.
    """
    overlay = frame.copy()

    # Draw left/right hands (same visual as Section 2, but no per-landmark depth)
    if frame_data['hands']['left']['joints']:
        overlay = draw_hand_landmarks(overlay, frame_data['hands']['left'], (0, 0, 255), camera_intrinsics, draw_depth=False)
    if frame_data['hands']['right']['joints']:
        overlay = draw_hand_landmarks(overlay, frame_data['hands']['right'], (255, 0, 0), camera_intrinsics, draw_depth=False)

    # Draw wrist depth only (millimeters) at the wrist 2D pixel coordinate for each hand
    for hand_key, color in [('left', (0, 0, 255)), ('right', (255, 0, 0))]:
        hand = frame_data['hands'][hand_key]
        if hand and hand.get('joints') and hand.get('joints_2d'):
            try:
                wrist_3d = hand['joints'][0]      # [x, y, z]
                wrist_px = hand['joints_2d'][0]  # [x_px, y_px]
                z_m = wrist_3d[2]
                depth_text = f"{z_m*1000:.0f}mm"

                x_px = int(wrist_px[0])
                y_px = int(wrist_px[1])

                # Draw a small filled background for legibility, then white text
                text_pos = (x_px + 8, y_px - 8)
                (tw, th), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(overlay, (text_pos[0]-4, text_pos[1]-th-4), (text_pos[0]+tw+4, text_pos[1]+4), (0,0,0), -1)
                cv2.putText(overlay, depth_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            except Exception:
                # keep overlay robust — if any field missing, skip
                pass

    # No additional text or legends — section3 is deliberately minimal
    return overlay

def create_section4(frame_data, width, height, show_hands=True):
    """Section 4: 3D matplotlib visualization.

    If `show_hands` is False the camera/head is shown but hand points are suppressed
    and an "Head unstable" label is added.
    """
    # Create matplotlib figure
    fig = Figure(figsize=(width/100, height/100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Hand Tracking 3D Visualization\n(Y is Up, XZ is Ground Plane)')
    
    # Camera/head position (green sphere)
    pose_matrix = np.array(frame_data['camera_pose_world'])
    camera_pos = pose_matrix[:3, 3]
    ax.scatter([camera_pos[0]], [camera_pos[1]], [camera_pos[2]], 
              c='green', marker='o', s=100, label='Camera')

    if show_hands:
        # Plot left hand (red)
        if frame_data['hands']['left']['joints']:
            left_joints = np.array(frame_data['hands']['left']['joints'])
            ax.scatter(left_joints[:, 0], left_joints[:, 1], left_joints[:, 2],
                      c='red', marker='o', s=20, label='Left Hand')
            # Draw connections
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(left_joints) and end_idx < len(left_joints):
                    points = np.array([left_joints[start_idx], left_joints[end_idx]])
                    ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'r-', linewidth=1)
            # wrist label
            wrist = left_joints[0]
            ax.text(wrist[0], wrist[1], wrist[2], f'  L wrist: {wrist[2]:.3f}m', fontsize=8, color='red')

        # Plot right hand (blue)
        if frame_data['hands']['right']['joints']:
            right_joints = np.array(frame_data['hands']['right']['joints'])
            ax.scatter(right_joints[:, 0], right_joints[:, 1], right_joints[:, 2],
                      c='blue', marker='o', s=20, label='Right Hand')
            # Draw connections
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(right_joints) and end_idx < len(right_joints):
                    points = np.array([right_joints[start_idx], right_joints[end_idx]])
                    ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=1)
            wrist = right_joints[0]
            ax.text(wrist[0], wrist[1], wrist[2], f'  R wrist: {wrist[2]:.3f}m', fontsize=8, color='blue')
    else:
        # Indicate head is unstable and hands are hidden
        ax.text(camera_pos[0], camera_pos[1], camera_pos[2], '  Head unstable — hiding hands', fontsize=10, color='orange')

    # Add legend
    ax.legend(loc='upper right', fontsize=8)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Set equal aspect ratio
    max_range = 0.5  # meters
    ax.set_xlim([camera_pos[0] - max_range, camera_pos[0] + max_range])
    ax.set_ylim([camera_pos[1] - max_range, camera_pos[1] + max_range])
    ax.set_zlim([camera_pos[2] - max_range, camera_pos[2] + max_range])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Convert to image
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    
    return img_bgr
def is_head_stable(frames_data, idx, window=HEAD_STABILITY_WINDOW, rot_thresh_deg=HEAD_ROT_THRESH_DEG):
    """Return True if head is stable over the last `window` frames.

    - Prefer `camera_orientation_fused` when available (radian angles, xyz).
    - Fallback: extract Euler from `camera_pose_world` rotation matrix.
    """
    start = max(0, idx - window + 1)
    angles = []
    for i in range(start, idx + 1):
        fd = frames_data[i]
        if 'camera_orientation_fused' in fd:
            ang = np.asarray(fd['camera_orientation_fused'], dtype=float)
            angles.append(ang)
        elif 'imu_orientation' in fd:
            ang = np.asarray(fd['imu_orientation'], dtype=float)
            angles.append(ang)
        else:
            try:
                Rmat = np.array(fd['camera_pose_world'])[:3, :3]
                ang = R.from_matrix(Rmat).as_euler('xyz', degrees=False)
                angles.append(ang)
            except Exception:
                # cannot evaluate — treat as unstable
                return False

    if len(angles) < 2:
        return False

    angles = np.asarray(angles)  # shape (k,3) in radians
    # unwrap to avoid discontinuities then compute frame-to-frame diffs
    angles_unwrapped = np.unwrap(angles, axis=0)
    diffs = np.abs(np.diff(angles_unwrapped, axis=0))  # radians
    diffs_deg = np.degrees(diffs)
    max_change = np.max(diffs_deg)
    return float(max_change) < float(rot_thresh_deg)

def combine_sections(sec1, sec2, sec3, sec4):
    """Combine 4 sections into 2x2 grid."""
    # Resize all sections to same size
    h, w = sec1.shape[:2]
    target_h, target_w = h // 2, w // 2
    
    sec1_small = cv2.resize(sec1, (target_w, target_h))
    sec2_small = cv2.resize(sec2, (target_w, target_h))
    sec3_small = cv2.resize(sec3, (target_w, target_h))
    sec4_small = cv2.resize(sec4, (target_w, target_h))
    
    # Add section labels
    cv2.putText(sec1_small, "1: Original Video", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(sec2_small, "2: Hand Tracking", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(sec3_small, "3: Depth Overlay", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(sec4_small, "4: 3D Visualization", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Combine into 2x2 grid
    top_row = np.hstack([sec1_small, sec2_small])
    bottom_row = np.hstack([sec3_small, sec4_small])
    combined = np.vstack([top_row, bottom_row])
    
    return combined

def generate_visualization():
    """Main function to generate 4-section visualization."""
    print("=" * 60)
    print("Generating 4-Section Comprehensive Visualization")
    print("=" * 60)
    
    # Load data
    frames_data, cap, fps, width, height, camera_intrinsics = load_data()
    
    # Setup output video
    os.makedirs('output', exist_ok=True)
    output_path = 'output/visualization_4section.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\nProcessing {len(frames_data)} frames...")
    print("This will take several minutes...\n")
    
    for idx, frame_data in enumerate(frames_data):
        # Read video frame
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {idx}")
            break
        
        # Generate all 4 sections
        section1 = create_section1(frame)
        section2 = create_section2(frame, frame_data, camera_intrinsics)
        section3 = create_section3(frame, frame_data, camera_intrinsics)

        # Head-stability check (use fused orientation when present)
        stable = is_head_stable(frames_data, idx, window=HEAD_STABILITY_WINDOW, rot_thresh_deg=HEAD_ROT_THRESH_DEG)
        section4 = create_section4(frame_data, width, height, show_hands=stable)
        
        # Combine sections
        final_frame = combine_sections(section1, section2, section3, section4)
        
        # Write frame
        out.write(final_frame)
        
        # Progress update
        if idx % 50 == 0:
            progress = (idx + 1) / len(frames_data) * 100
            print(f"  Progress: {idx+1}/{len(frames_data)} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n✓ Video saved to: {output_path}")
    
    # Show file info
    import subprocess
    result = subprocess.run(['ls', '-lh', output_path], capture_output=True, text=True)
    print(result.stdout)
    
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 
                           'format=duration', '-of', 
                           'default=noprint_wrappers=1:nokey=1', output_path],
                          capture_output=True, text=True)
    if result.returncode == 0:
        duration = float(result.stdout.strip())
        print(f"Duration: {duration:.2f} seconds")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    generate_visualization()
