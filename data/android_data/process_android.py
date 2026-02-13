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
import argparse
import time

# Configuration defaults
DEFAULT_CAMERA_INTRINSICS = {'fx': 500, 'fy': 500, 'cx': 320, 'cy': 240, 'width': 640, 'height': 480}
DEFAULT_SAMPLE_RATE = 1
DEFAULT_DEPTH_DOWNSCALE = 1
USE_GPU_BY_DEFAULT = True

def load_data():
    with open('motion_data.json') as f:
        motion = json.load(f)
    with open('video_timestamps.json') as f:
        timestamps = json.load(f)['timestamps']
    cap = cv2.VideoCapture('video.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return motion, timestamps, cap, fps, total_frames

def initialize_models(use_cuda: bool = USE_GPU_BY_DEFAULT):
    """Load MiDaS and MediaPipe models and return (midas, transforms, hands, camera_matrix, device).
    - Moves MiDaS to the selected device so inference can run on GPU when available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    # MiDaS (moved to device)
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    # MediaPipe hand landmarker
    base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    hands = vision.HandLandmarker.create_from_options(options)

    cm = DEFAULT_CAMERA_INTRINSICS
    camera_matrix = np.array([[cm['fx'], 0, cm['cx']], [0, cm['fy'], cm['cy']], [0, 0, 1]])

    return midas, midas_transforms, hands, camera_matrix, device

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

def estimate_depth(midas, midas_transforms, rgb, device=None, downscale: int = 1, timing_list=None):
    """Run MiDaS on `rgb` and return a depth map at the original image resolution.

    Optionally records per-inference timings to `timing_list` (append, seconds).

    Args:
        device: torch device where `midas` is located (recommended).
        downscale: integer factor to downsample input before MiDaS (faster, lower res).
        timing_list: optional list to append per-inference elapsed seconds to.
    """
    # determine device
    if device is None:
        try:
            device = next(midas.parameters()).device
        except Exception:
            device = torch.device('cpu')

    # optionally downscale input for speed
    if downscale != 1 and downscale > 0:
        small = cv2.resize(rgb, (rgb.shape[1] // downscale, rgb.shape[0] // downscale))
        input_batch = midas_transforms(small).to(device)
    else:
        input_batch = midas_transforms(rgb).to(device)

    # Measure inference time (accurate on CUDA with torch.cuda.synchronize)
    start = None
    end = None
    try:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            depth_pred = midas(input_batch).squeeze().cpu().numpy()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
    except Exception:
        # fallback: run without timing if something goes wrong
        with torch.no_grad():
            depth_pred = midas(input_batch).squeeze().cpu().numpy()

    if start is not None and end is not None and timing_list is not None:
        timing_list.append(end - start)

    # upsample back to original image size when downscaled
    if downscale != 1 and downscale > 0:
        depth_pred = cv2.resize(depth_pred, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    return depth_pred

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


def process_single_frame(frame_bgr, timestamp, hands_model, midas, midas_transforms, camera_matrix, device=None, depth_downscale=1, timing_list=None):
    """Run detection + depth + lift for a single BGR frame and return `hand_data`."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    hand_data = detect_hands(hands_model, rgb)
    depth = estimate_depth(midas, midas_transforms, rgb, device=device, downscale=depth_downscale, timing_list=timing_list)
    hand_data = lift_hands_to_3d(hand_data, depth, camera_matrix, image_shape=rgb.shape[:2])
    return hand_data


def process_frames(cap, timestamps, hands, midas, midas_transforms, camera_matrix, total_frames, device=None, sample_rate: int = DEFAULT_SAMPLE_RATE, depth_downscale: int = DEFAULT_DEPTH_DOWNSCALE, imu_poses=None):
    """Process video frames with hand tracking, depth estimation and pose tracking.

    New parameters:
      - device: torch device where MiDaS runs
      - sample_rate: process every Nth frame (1 == every frame)
      - depth_downscale: MiDaS downscale factor for faster inference
    """
    frames = []
    frame_idx = 0
    current_pose = np.eye(4)
    prev_frame = None
    prev_features = None

    # collect MiDaS timings per processed frame
    midas_timings = []

    # Process every Nth frame for efficiency (sample rate)
    sample_rate = max(1, int(sample_rate))

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

        # Visual odometry for camera pose estimation
        current_pose = track_features(prev_frame, gray, prev_features, camera_matrix, current_pose)
        prev_features = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
        prev_frame = gray

        # Detection + depth + lift (modular helper) -- collect MiDaS timing
        hand_data = process_single_frame(frame, timestamps[frame_idx], hands, midas, midas_transforms, camera_matrix, device=device, depth_downscale=depth_downscale, timing_list=midas_timings)

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
            # report midas average so far
            if len(midas_timings) > 0:
                avg_midas_ms = float(np.mean(midas_timings)) * 1000.0
                print(f"  Processed frame {frame_idx}/{total_frames} — MiDaS avg: {avg_midas_ms:.1f} ms/frame")
            else:
                print(f"  Processed frame {frame_idx}/{total_frames}")

        frame_idx += 1

    cap.release()

    # final MiDaS timing summary
    if len(midas_timings) > 0:
        avg_midas_ms = float(np.mean(midas_timings)) * 1000.0
        p50 = float(np.percentile(midas_timings, 50)) * 1000.0
        p95 = float(np.percentile(midas_timings, 95)) * 1000.0
        print(f"Processing complete: {len(frames)} frames processed. MiDaS avg={avg_midas_ms:.1f}ms (p50={p50:.1f}ms p95={p95:.1f}ms) over {len(midas_timings)} samples")
    else:
        print(f"Processing complete: {len(frames)} frames processed. No MiDaS timing samples recorded.")

    return frames
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
    """Write Section‑3 visualization (MediaPipe overlay + wrist depth only).

    - Reads `video.mp4` frames in lock-step with `frames`.
    - Draws hand connections/landmarks using stored `joints_2d` when available.
    - Displays **only** wrist depth (mm) at the wrist pixel for each detected hand.
    - Writes `output/visualization_android.mp4` (overwrites previous file).
    """
    os.makedirs('output', exist_ok=True)
    video_path = 'video.mp4'
    out_path = 'output/visualization_android.mp4'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Warning: cannot open video file for visualization.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]

    print(f"   - Writing Section-3 visualization to: {out_path} ({len(frames)} frames)")

    for idx, frame_data in enumerate(frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: couldn't read frame {idx} from video file, stopping.")
            break

        def _get_2d_points(hand):
            if not hand or not hand.get('joints'):
                return []
            if hand.get('joints_2d'):
                return [ (int(x), int(y)) for x, y in hand['joints_2d'] ]
            # fallback: simple projection using hardcoded intrinsics
            fx, fy, cx, cy = 500, 500, 320, 240
            pts = []
            for x3, y3, z3 in hand['joints']:
                if z3 > 0:
                    x_px = int(x3 * fx / z3 + cx)
                    y_px = int(y3 * fy / z3 + cy)
                else:
                    x_px, y_px = cx, cy
                x_px = max(0, min(frame.shape[1]-1, x_px))
                y_px = max(0, min(frame.shape[0]-1, y_px))
                pts.append((x_px, y_px))
            return pts

        # Draw left (red) and right (blue)
        for hand_key, color in [('left', (0,0,255)), ('right', (255,0,0))]:
            hand = frame_data['hands'].get(hand_key, {})
            pts = _get_2d_points(hand)
            # connections
            for s, e in HAND_CONNECTIONS:
                if s < len(pts) and e < len(pts):
                    cv2.line(frame, pts[s], pts[e], color, 2)
            # landmarks
            for (x, y) in pts:
                cv2.circle(frame, (x, y), 4, color, -1)
                cv2.circle(frame, (x, y), 5, (255,255,255), 1)

        # Wrist depth only (displayed on wrist pixel) — left & right
        for hand_key, color in [('left', (0,0,255)), ('right', (255,0,0))]:
            hand = frame_data['hands'].get(hand_key)
            if hand and hand.get('joints') and hand.get('joints_2d'):
                try:
                    wrist_3d = hand['joints'][0]
                    wrist_px = hand['joints_2d'][0]
                    z_m = wrist_3d[2]
                    depth_text = f"{z_m*1000:.0f}mm"
                    x_px = int(wrist_px[0]); y_px = int(wrist_px[1])
                    text_pos = (x_px + 8, y_px - 8)
                    (tw, th), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(frame, (text_pos[0]-4, text_pos[1]-th-4),
                                  (text_pos[0]+tw+4, text_pos[1]+4), (0,0,0), -1)
                    cv2.putText(frame, depth_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                except Exception:
                    pass

        out.write(frame)

        if idx % 50 == 0:
            print(f"     Written {idx+1}/{len(frames)} frames")

    cap.release()
    out.release()
    print(f"   - ✓ Section‑3 video saved to: {out_path}")

def main():
    print("=" * 60)
    print("Android Hand Tracking Data Processing")
    print("=" * 60)
    
    print("\n1. Loading data...")
    motion, timestamps, cap, fps, total_frames = load_data()
    print(f"   - Video: {total_frames} frames at {fps} FPS")
    print(f"   - Timestamps: {len(timestamps)} entries")
    print(f"   - IMU samples: {motion['metadata']['sample_count']}")
    
    # Parse CLI args early so they can control behavior below
    parser = argparse.ArgumentParser(description='Process Android hand-tracking dataset')
    parser.add_argument('--max-frames', type=int, default=0, help='Limit number of frames to process (0 = all)')
    parser.add_argument('--sample-rate', type=int, default=DEFAULT_SAMPLE_RATE, help='Process every Nth frame')
    parser.add_argument('--depth-downscale', type=int, default=DEFAULT_DEPTH_DOWNSCALE, help='Downscale factor for MiDaS input')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU even if available')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage (error if not available)')
    args = parser.parse_args()

    # Decide whether to use CUDA: --gpu explicitly requests it, otherwise use_cuda unless --no-gpu set
    if args.gpu and not torch.cuda.is_available():
        raise SystemExit("Requested --gpu but CUDA is not available in this Python environment.")
    use_cuda = args.gpu or (not args.no_gpu)

    # Process frames (may be limited via CLI args)
    max_frames = args.max_frames if (hasattr(args, 'max_frames') and args.max_frames > 0) else total_frames
    print(f"   - Processing {max_frames} frames (sample_rate={args.sample_rate}, depth_downscale={args.depth_downscale})")

    midas, midas_transforms, hands, camera_matrix, device = initialize_models(use_cuda=use_cuda)
    print(f"   - MiDaS depth estimator loaded (device={device})")

    # If running on CUDA, print the GPU name for clarity
    if str(device).startswith('cuda') and torch.cuda.is_available():
        try:
            dev_idx = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(dev_idx)
        except Exception:
            dev_name = str(device)
        print(f"   - CUDA device: {dev_name}")

    print("   - MediaPipe hand tracker loaded")

    # Quick MiDaS warm-up (small number of runs) to estimate per-frame inference time
    try:
        w = DEFAULT_CAMERA_INTRINSICS['width']
        h = DEFAULT_CAMERA_INTRINSICS['height']
        dummy_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        input_batch = midas_transforms(dummy_rgb).to(device)
        # single warm-up + timed runs
        with torch.no_grad():
            if device.type == 'cuda':
                torch.cuda.synchronize()
            _ = midas(input_batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        times = []
        runs = 3
        with torch.no_grad():
            for _ in range(runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = midas(input_batch)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)
        avg_ms = (sum(times) / len(times)) * 1000.0
        print(f"   - MiDaS warm-up avg inference: {avg_ms:.1f} ms/frame ({runs} runs)")
    except Exception as e:
        print(f"   - MiDaS warm-up failed: {e}")
    
    print("\n3. Processing IMU data...")
    imu_timestamps, orientation, position = process_imu(motion)
    print(f"   - Computed poses for {len(imu_timestamps)} IMU samples")
    print("   - Note: IMU available for future fusion with visual odometry")
    
    print("\n4. Processing video frames...")
    frames = process_frames(cap, timestamps, hands, midas, midas_transforms, camera_matrix, max_frames, device=device, sample_rate=args.sample_rate, depth_downscale=args.depth_downscale)
    
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