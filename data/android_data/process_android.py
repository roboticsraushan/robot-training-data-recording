import json
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import torch
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import os
from PIL import Image
import argparse
import time
import sys

# visualization imports (4-section visualizer)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D

# Configuration defaults
DEFAULT_CAMERA_INTRINSICS = {'fx': 500, 'fy': 500, 'cx': 320, 'cy': 240, 'width': 640, 'height': 480}
DEFAULT_SAMPLE_RATE = 1
DEFAULT_DEPTH_DOWNSCALE = 1
USE_GPU_BY_DEFAULT = True
# Nominal scene distance (used to convert MiDaS relative depths → approximate meters)
DEFAULT_NOMINAL_DEPTH_M = 1.0  # meters (adjustable)

# Visualization / head-stability defaults
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]
HEAD_STABILITY_WINDOW = 3       # frames
HEAD_ROT_THRESH_DEG = 5.0       # degrees (lenient)

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

    # If IMU timestamps appear to be negative (device-relative), align them to
    # the recording_date epoch (if available) so they share the same timebase
    # as video timestamps (which are session timestamps).
    try:
        meta = motion.get('metadata', {})
        rec_date = meta.get('recording_date')
        if rec_date and np.median(imu_timestamps) < 0:
            from datetime import datetime, timezone
            epoch = float(datetime.fromisoformat(rec_date.replace('Z', '+00:00')).timestamp())
            imu_timestamps = imu_timestamps + epoch
            print(f"   - Aligned IMU timestamps to recording_date epoch: {epoch} (converted to session timebase)")
    except Exception:
        # if parsing fails, leave imu_timestamps unchanged
        pass

    # Prefer euler angles (roll, pitch, yaw) if present in the recorded columns
    try:
        # motion['metadata']['columns'] indicates indices; roll,pitch,yaw are expected at indices 14..16
        euler = np.array([e[14:17] for e in data], dtype=float)
        orientation = euler  # already in radians per metadata
    except Exception:
        # fallback: integrate gyro (rad/s) to get orientation
        gyro = np.array([e[1:4] for e in data])
        dt = np.diff(imu_timestamps, prepend=imu_timestamps[0])
        orientation = np.cumsum(gyro * dt[:, None], axis=0)

    # compute simple velocity/position from accel (kept for completeness)
    try:
        accel = np.array([e[4:7] for e in data])
        velocity = np.array([cumulative_trapezoid(accel[:, i], imu_timestamps, initial=0) for i in range(3)]).T
        position = np.array([cumulative_trapezoid(velocity[:, i], imu_timestamps, initial=0) for i in range(3)]).T
    except Exception:
        velocity = np.zeros((len(imu_timestamps), 3))
        position = np.zeros((len(imu_timestamps), 3))

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

def estimate_depth(midas, midas_transforms, rgb, device=None, downscale: int = 1, timing_list=None, nominal_depth_m: float = DEFAULT_NOMINAL_DEPTH_M):
    """Run MiDaS on `rgb` and return a depth map and an adaptive metric scale.

    Returns:
      (depth_pred, depth_scale)
      - depth_pred: MiDaS output (same units as before)
      - depth_scale: multiplicative factor so that z_m = (1.0 / depth_pred) * depth_scale

    The scale is computed per-frame by mapping the median of the *relative* depths
    (1 / depth_pred) in the central image crop to `nominal_depth_m`.

    Args:
        device: torch device where `midas` is located (recommended).
        downscale: integer factor to downsample input before MiDaS (faster, lower res).
        timing_list: optional list to append per-inference elapsed seconds to.
        nominal_depth_m: the scene distance (meters) that the median relative depth maps to.
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

    # --- compute adaptive metric scale for this frame ---
    # MiDaS output can be treated as an 'inverse-depth-like' quantity; we compute
    # inv = 1 / depth_pred and map its median to nominal_depth_m.
    eps = 1e-6
    depth_safe = np.clip(depth_pred, eps, None)
    inv_map = 1.0 / depth_safe

    # use a central crop to avoid sky/background outliers
    h, w = inv_map.shape[:2]
    y0, y1 = int(h * 0.25), int(h * 0.75)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    crop = inv_map[y0:y1, x0:x1]
    median_inv = float(np.median(crop)) if crop.size > 0 else float(np.median(inv_map))
    depth_scale = nominal_depth_m / (median_inv + eps)

    return depth_pred, depth_scale

def lift_hands_to_3d(hand_data, depth, camera_matrix, image_shape=None, depth_scale: float = None):
    """Lift 2D hand landmarks to 3D using depth map.
    - Convert MediaPipe normalized coords to *image* pixel coordinates (for correct overlay).
    - Sample the MiDaS depth map by mapping image pixels into depth-map coordinates.
    - Unproject using camera intrinsics (which expect image pixels).

    Args:
        image_shape: optional (height, width) of the original RGB frame. If omitted,
                     depth.shape will be used as a fallback (but that may cause misalignment).
        depth_scale: optional per-frame scale so that final z_m = (1.0/depth_value) * depth_scale.
                     If omitted, falls back to previous empirical mapping (0.5 / depth_value).
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

                # Convert MiDaS output to metric depth.
                # If a per-frame depth_scale was provided use:
                #    z = (1.0 / depth_value) * depth_scale
                # Otherwise fall back to the older empirical constant mapping.
                eps = 1e-6
                if depth_scale is not None:
                    inv = 1.0 / (depth_value + eps)
                    z = float(inv * depth_scale)
                else:
                    z = 0.5 / (depth_value + eps)

                # Unproject using IMAGE pixel coords (camera intrinsics are image-based)
                x_3d = (x_px_img - cx) * z / fx
                y_3d = (y_px_img - cy) * z / fy
                landmarks_3d.append([x_3d, y_3d, z])

            hand_data[hand]['landmarks_3d'] = np.array(landmarks_3d)
            hand_data[hand]['landmarks_2d_pixels'] = np.array(landmarks_2d_pixels)
    return hand_data


def process_single_frame(frame_bgr, timestamp, hands_model, midas, midas_transforms, camera_matrix, device=None, depth_downscale=1, timing_list=None, nominal_depth_m: float = DEFAULT_NOMINAL_DEPTH_M):
    """Run detection + depth + lift for a single BGR frame and return `hand_data`."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    hand_data = detect_hands(hands_model, rgb)
    depth_pred, depth_scale = estimate_depth(midas, midas_transforms, rgb, device=device, downscale=depth_downscale, timing_list=timing_list, nominal_depth_m=nominal_depth_m)
    hand_data = lift_hands_to_3d(hand_data, depth_pred, camera_matrix, image_shape=rgb.shape[:2], depth_scale=depth_scale)
    return hand_data


def process_frames(cap, timestamps, hands, midas, midas_transforms, camera_matrix, total_frames, device=None, sample_rate: int = DEFAULT_SAMPLE_RATE, depth_downscale: int = DEFAULT_DEPTH_DOWNSCALE, nominal_depth_m: float = DEFAULT_NOMINAL_DEPTH_M, imu_poses=None):
    """Process video frames with hand tracking, depth estimation and pose tracking.

    New parameters:
      - device: torch device where MiDaS runs
      - sample_rate: process every Nth frame (1 == every frame)
      - depth_downscale: MiDaS downscale factor for faster inference
      - imu_poses: optional tuple (imu_timestamps, imu_orientations) for IMU/VO fusion
    """
    frames = []
    frame_idx = 0
    current_pose = np.eye(4)
    prev_frame = None
    prev_features = None

    # collect MiDaS timings per processed frame
    midas_timings = []

    # IMU arrays (if provided)
    imu_ts = None
    imu_orients = None
    if imu_poses is not None:
        try:
            imu_ts, imu_orients = imu_poses
            imu_ts = np.asarray(imu_ts)
            imu_orients = np.asarray(imu_orients)
        except Exception:
            imu_ts, imu_orients = None, None

    # fusion weight (vo contribution when fusing angles)
    # VO disabled per user request — rely on IMU only for head tracking
    VO_WEIGHT = 0.0  # visual-odometry contribution turned off (use IMU-only)

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

        # Visual odometry disabled — using IMU-only head tracking (track_features commented out)
        # current_pose = track_features(prev_frame, gray, prev_features, camera_matrix, current_pose)
        # prev_features = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
        # prev_frame = gray

        # Detection + depth + lift (modular helper) -- collect MiDaS timing
        hand_data = process_single_frame(frame, timestamps[frame_idx], hands, midas, midas_transforms, camera_matrix, device=device, depth_downscale=depth_downscale, timing_list=midas_timings, nominal_depth_m=nominal_depth_m)

        t = timestamps[frame_idx]

        # Interpolate IMU orientation at this frame timestamp (if available)
        imu_interp_angles = None
        if imu_ts is not None and imu_orients is not None and imu_ts.size > 0:
            # Only interpolate when the video frame timestamp falls inside the IMU time range —
            # avoid using the nearest-edge IMU sample for the whole video which incorrectly
            # shows a static orientation when IMU only covers a short tail.
            if (t >= float(imu_ts[0])) and (t <= float(imu_ts[-1])):
                try:
                    imu_interp_angles = np.array([
                        float(np.interp(t, imu_ts, imu_orients[:, 0])),
                        float(np.interp(t, imu_ts, imu_orients[:, 1])),
                        float(np.interp(t, imu_ts, imu_orients[:, 2]))
                    ])
                except Exception:
                    imu_interp_angles = None
            else:
                # IMU has no sample for this frame timestamp — leave as None
                imu_interp_angles = None

        # VO disabled — do not compute vo_angles from visual odometry; rely on IMU only
        vo_angles = None

        # Loose-coupled fusion (slerp in Euler-space approximation)
        fused_angles = None
        if imu_interp_angles is not None and vo_angles is not None:
            # Both available: fuse with simple weighted average on angles (small VO correction)
            fused_angles = (1.0 - VO_WEIGHT) * imu_interp_angles + VO_WEIGHT * vo_angles
        elif imu_interp_angles is not None:
            fused_angles = imu_interp_angles
        elif vo_angles is not None:
            fused_angles = vo_angles

        # If we have a fused / IMU orientation, apply it to the current camera pose's rotation
        # (this maps the IMU-on-head orientation into camera_pose_world so the head/camera will rotate in the viz).
        if fused_angles is not None:
            try:
                # fused_angles are stored as Euler angles in radians (xyz)
                Rmat = R.from_euler('xyz', fused_angles, degrees=False).as_matrix()
                current_pose[:3, :3] = Rmat
            except Exception:
                pass

        # Build frame entry
        frame_entry = {
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
        }

        # attach IMU/VO/fused orientations for later visualization/fusion
        if imu_interp_angles is not None:
            frame_entry['imu_orientation'] = imu_interp_angles.tolist()
        if vo_angles is not None:
            frame_entry['vo_orientation'] = vo_angles.tolist()
        if fused_angles is not None:
            frame_entry['camera_orientation_fused'] = fused_angles.tolist()

        frames.append(frame_entry)

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
        depth_pred, depth_scale = estimate_depth(midas, midas_transforms, rgb, nominal_depth_m=DEFAULT_NOMINAL_DEPTH_M)
        # Pass the original image size so 2D pixel coords are computed in IMAGE space
        hand_data = lift_hands_to_3d(hand_data, depth_pred, camera_matrix, image_shape=rgb.shape[:2], depth_scale=depth_scale)
        
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


def validate_unified_json(json_path='output/unified_hand_tracking_android.json', schema_path='schema/unified_hand_tracking_schema.json'):
    """Validate the produced JSON against the repository schema.
    Returns True on success or when validation is skipped (no jsonschema available).
    Returns False only when validation ran and found schema violations.
    """
    try:
        from jsonschema import Draft7Validator
    except Exception:
        print("   - JSON schema validation skipped: 'jsonschema' not installed (pip install jsonschema)")
        return True

    # ensure files exist
    if not os.path.exists(json_path):
        print(f"   - JSON validation skipped: file not found: {json_path}")
        return False
    if not os.path.exists(schema_path):
        print(f"   - JSON validation skipped: schema not found: {schema_path}")
        return False

    with open(schema_path, 'r') as sf:
        schema = json.load(sf)
    with open(json_path, 'r') as jf:
        data = json.load(jf)

    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(data))
    if not errors:
        print('   - JSON schema validation: SUCCESS')
        return True

    print(f'   - JSON schema validation: FAILED ({len(errors)} errors)')
    for i, e in enumerate(sorted(errors, key=lambda x: x.path)):
        path = '.'.join(map(str, e.absolute_path)) if e.absolute_path else '<root>'
        print(f'     {i+1}) path: {path}\n         message: {e.message}')
    return False

def generate_visualization(frames, section='3'):
    """Flexible visualizer — supports sections 1..4 or the full 2x2 composite ('all').

    Args:
        frames: list of frame entries produced by `process_frames` / saved JSON
        section: '1'|'2'|'3'|'4'|'all' — which section to write (default '3')
    """
    # helper: draw hand landmarks + optional per-landmark depth
    def draw_hand_landmarks(img, hand_data, color, camera_intrinsics, draw_depth=False):
        if not hand_data or hand_data.get('confidence', 0) == 0 or not hand_data.get('joints'):
            return img

        # prefer stored 2D pixel coordinates when available
        if 'joints_2d' in hand_data and hand_data['joints_2d']:
            landmarks_2d = [(int(x), int(y)) for x, y in hand_data['joints_2d']]
        else:
            landmarks_2d = []
            fx = camera_intrinsics['fx']
            fy = camera_intrinsics['fy']
            cx = camera_intrinsics['cx']
            cy = camera_intrinsics['cy']
            for x3, y3, z3 in hand_data['joints']:
                if z3 > 0:
                    x_px = int(x3 * fx / z3 + cx)
                    y_px = int(y3 * fy / z3 + cy)
                else:
                    x_px, y_px = int(cx), int(cy)
                x_px = max(0, min(img.shape[1] - 1, x_px))
                y_px = max(0, min(img.shape[0] - 1, y_px))
                landmarks_2d.append((x_px, y_px))

        # draw connections
        for s, e in HAND_CONNECTIONS:
            if s < len(landmarks_2d) and e < len(landmarks_2d):
                cv2.line(img, landmarks_2d[s], landmarks_2d[e], color, 2)

        # draw landmarks and optional depth
        for idx, (x, y) in enumerate(landmarks_2d):
            cv2.circle(img, (x, y), 4, color, -1)
            cv2.circle(img, (x, y), 5, (255, 255, 255), 1)
            if draw_depth and idx < len(hand_data.get('joints', [])):
                depth_m = hand_data['joints'][idx][2]
                cv2.putText(img, f"{depth_m*1000:.0f}mm", (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

        return img

    def create_section1(frame, frame_data, camera_intrinsics):
        return frame.copy()

    def create_section2(frame, frame_data, camera_intrinsics):
        overlay = frame.copy()
        if frame_data['hands']['left']['joints']:
            overlay = draw_hand_landmarks(overlay, frame_data['hands']['left'], (0,0,255), camera_intrinsics, draw_depth=False)
        if frame_data['hands']['right']['joints']:
            overlay = draw_hand_landmarks(overlay, frame_data['hands']['right'], (255,0,0), camera_intrinsics, draw_depth=False)
        cv2.putText(overlay, "MediaPipe Hand Tracking", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return overlay

    def create_section3(frame, frame_data, camera_intrinsics):
        overlay = frame.copy()
        # draw hand skeletons but only show wrist depth text
        if frame_data['hands']['left']['joints']:
            overlay = draw_hand_landmarks(overlay, frame_data['hands']['left'], (0,0,255), camera_intrinsics, draw_depth=False)
        if frame_data['hands']['right']['joints']:
            overlay = draw_hand_landmarks(overlay, frame_data['hands']['right'], (255,0,0), camera_intrinsics, draw_depth=False)
        for hand_key, color in [('left',(0,0,255)),('right',(255,0,0))]:
            hand = frame_data['hands'][hand_key]
            if hand and hand.get('joints') and hand.get('joints_2d'):
                try:
                    wrist_3d = hand['joints'][0]
                    wrist_px = hand['joints_2d'][0]
                    z_m = wrist_3d[2]
                    depth_text = f"{z_m*1000:.0f}mm"
                    x_px = int(wrist_px[0]); y_px = int(wrist_px[1])
                    text_pos = (x_px + 8, y_px - 8)
                    (tw, th), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(overlay, (text_pos[0]-4, text_pos[1]-th-4), (text_pos[0]+tw+4, text_pos[1]+4), (0,0,0), -1)
                    cv2.putText(overlay, depth_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                except Exception:
                    pass
        return overlay

    def create_section4(frame_data, width, height, show_hands=True, scene_bounds=None):
        """Render 3D view with Z as the vertical axis (Z = up). Data Z is flipped
        when visualizing so positive Z points upward in the renderer.
        """
        fig = Figure(figsize=(width/100, height/100), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Hand Tracking 3D Visualization\n(Z is Up, XY is Ground Plane)')

        pose_matrix = np.array(frame_data['camera_pose_world'])
        camera_pos = pose_matrix[:3, 3]
        # flip Z for visualization so Z points up
        cam_v = np.array([camera_pos[0], camera_pos[1], -camera_pos[2]])
        ax.scatter([cam_v[0]], [cam_v[1]], [cam_v[2]], c='green', marker='o', s=80, label='Camera')

        # --- draw camera / head orientation axes derived from IMU or fused orientation ---
        # Priority: use fused orientation -> imu_orientation -> pose rotation
        try:
            if 'camera_orientation_fused' in frame_data:
                rot = R.from_euler('xyz', np.asarray(frame_data['camera_orientation_fused'], dtype=float), degrees=False).as_matrix()
            elif 'imu_orientation' in frame_data:
                rot = R.from_euler('xyz', np.asarray(frame_data['imu_orientation'], dtype=float), degrees=False).as_matrix()
            else:
                rot = pose_matrix[:3, :3]

            # Axis length for visual triad (meters)
            axis_len = max(0.08, (scene_bounds[3] - scene_bounds[0]) * 0.05) if scene_bounds is not None else 0.08
            # Draw X (red), Y (green), Z (blue) — transform Z for Z-up visualization
            axes_colors = [('X', 'r'), ('Y', 'g'), ('Z', 'b')]
            for i, (label, col) in enumerate(axes_colors):
                vec = rot[:, i]
                viz_vec = np.array([vec[0], vec[1], -vec[2]])  # flip Z for viz
                end = cam_v + axis_len * viz_vec
                ax.plot([cam_v[0], end[0]], [cam_v[1], end[1]], [cam_v[2], end[2]], color=col, linewidth=2)
                ax.text(end[0], end[1], end[2], f' {label}', color=col, fontsize=8)
        except Exception:
            # if orientation parsing fails, continue without axes
            pass

        if show_hands:
            if frame_data['hands']['left']['joints']:
                left_joints = np.array(frame_data['hands']['left']['joints'])
                left_v = left_joints.copy()
                left_v[:, 2] = -left_v[:, 2]
                ax.scatter(left_v[:,0], left_v[:,1], left_v[:,2], c='red', s=20)
                for s_idx, e_idx in HAND_CONNECTIONS:
                    if s_idx < len(left_v) and e_idx < len(left_v):
                        pts = np.vstack([left_v[s_idx], left_v[e_idx]])
                        ax.plot3D(pts[:,0], pts[:,1], pts[:,2], 'r-')
                wrist = left_v[0]
                ax.text(wrist[0], wrist[1], wrist[2], f'  L wrist: {wrist[2]:.3f}m', fontsize=8, color='red')
            if frame_data['hands']['right']['joints']:
                right_joints = np.array(frame_data['hands']['right']['joints'])
                right_v = right_joints.copy()
                right_v[:, 2] = -right_v[:, 2]
                ax.scatter(right_v[:,0], right_v[:,1], right_v[:,2], c='blue', s=20)
                for s_idx, e_idx in HAND_CONNECTIONS:
                    if s_idx < len(right_v) and e_idx < len(right_v):
                        pts = np.vstack([right_v[s_idx], right_v[e_idx]])
                        ax.plot3D(pts[:,0], pts[:,1], pts[:,2], 'b-')
                wrist = right_v[0]
                ax.text(wrist[0], wrist[1], wrist[2], f'  R wrist: {wrist[2]:.3f}m', fontsize=8, color='blue')
        else:
            ax.text(cam_v[0], cam_v[1], cam_v[2], '  Head unstable — hiding hands', fontsize=10, color='orange')

        ax.legend(loc='upper right', fontsize=8)
        ax.view_init(elev=20, azim=45)

        # Use provided scene_bounds (stable axes) if available, otherwise fallback to camera-centric box
        if scene_bounds is not None:
            xmin, ymin, zmin, xmax, ymax, zmax = scene_bounds
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_zlim([zmin, zmax])
        else:
            pad = 0.5
            ax.set_xlim([cam_v[0]-pad, cam_v[0]+pad])
            ax.set_ylim([cam_v[1]-pad, cam_v[1]+pad])
            ax.set_zlim([cam_v[2]-pad, cam_v[2]+pad])

        ax.grid(True, alpha=0.3)

        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return img_bgr

    def is_head_stable(frames_data, idx, window=HEAD_STABILITY_WINDOW, rot_thresh_deg=HEAD_ROT_THRESH_DEG):
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
                    return False
        if len(angles) < 2:
            return False
        angles = np.asarray(angles)
        angles_unwrapped = np.unwrap(angles, axis=0)
        diffs = np.abs(np.diff(angles_unwrapped, axis=0))
        diffs_deg = np.degrees(diffs)
        max_change = np.max(diffs_deg)
        return float(max_change) < float(rot_thresh_deg)

    def combine_sections(sec1, sec2, sec3, sec4):
        h, w = sec1.shape[:2]
        target_h, target_w = h//2, w//2
        s1 = cv2.resize(sec1, (target_w, target_h))
        s2 = cv2.resize(sec2, (target_w, target_h))
        s3 = cv2.resize(sec3, (target_w, target_h))
        s4 = cv2.resize(sec4, (target_w, target_h))
        cv2.putText(s1, "1: Original Video", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(s2, "2: Hand Tracking", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(s3, "3: Depth Overlay", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(s4, "4: 3D Visualization", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        top = np.hstack([s1, s2])
        bot = np.hstack([s3, s4])
        return np.vstack([top, bot])

    # --- open video and prepare writer ---
    os.makedirs('output', exist_ok=True)
    video_path = 'video.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Warning: cannot open video file for visualization.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # decide output filename based on requested section
    if section == 'all':
        out_path = 'output/visualization_4section.mp4'
        out_size = (width, height)
    elif section == '3':
        out_path = 'output/visualization_android.mp4'
        out_size = (width, height)
    else:
        out_path = f'output/visualization_section{section}.mp4'
        out_size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, out_size)

    print(f"   - Writing visualization (section={section}) to: {out_path} ({len(frames)} frames)")

    # use camera intrinsics from module default for projections when needed
    camera_intrinsics = DEFAULT_CAMERA_INTRINSICS

    # --- compute global scene bounds so Section-4 axes remain fixed across frames ---
    # Re-orient all scene points into the Z-up visualization frame used by create_section4
    all_pts = []
    for f in frames:
        try:
            cam = np.array(f['camera_pose_world'])[:3, 3]
            all_pts.append(_to_viz_zup(cam))
        except Exception:
            pass
        for hk in ('left', 'right'):
            h = f['hands'].get(hk, {})
            if h and h.get('joints'):
                try:
                    for j in h['joints']:
                        all_pts.append(_to_viz_zup(j))
                except Exception:
                    pass

    if len(all_pts) == 0:
        # fallback bounds around origin
        scene_min = np.array([-0.5, -0.5, -0.5])
        scene_max = np.array([0.5, 0.5, 0.5])
    else:
        arr = np.asarray(all_pts, dtype=float)
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        center = (mins + maxs) / 2.0
        max_range = np.max(maxs - mins)
        pad = max(0.5, max_range * 0.6)
        scene_min = center - pad
        scene_max = center + pad

    scene_bounds = (float(scene_min[0]), float(scene_min[1]), float(scene_min[2]),
                    float(scene_max[0]), float(scene_max[1]), float(scene_max[2]))

    for idx, frame_data in enumerate(frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: couldn't read frame {idx} from video file, stopping.")
            break

        if section == 'all':
            s1 = create_section1(frame, frame_data, camera_intrinsics)
            s2 = create_section2(frame, frame_data, camera_intrinsics)
            s3 = create_section3(frame, frame_data, camera_intrinsics)
            stable = is_head_stable(frames, idx)
            s4 = create_section4(frame_data, width, height, show_hands=stable, scene_bounds=scene_bounds)
            final = combine_sections(s1, s2, s3, s4)
        elif section == '1':
            final = create_section1(frame, frame_data, camera_intrinsics)
        elif section == '2':
            final = create_section2(frame, frame_data, camera_intrinsics)
        elif section == '3':
            final = create_section3(frame, frame_data, camera_intrinsics)
        elif section == '4':
            stable = is_head_stable(frames, idx)
            final = create_section4(frame_data, width, height, show_hands=stable, scene_bounds=scene_bounds)
        else:
            print(f"Unknown section: {section}")
            break

        # ensure final frame matches writer size
        if final.shape[1] != out_size[0] or final.shape[0] != out_size[1]:
            final = cv2.resize(final, out_size)

        out.write(final)

        if idx % 50 == 0:
            print(f"     Written {idx+1}/{len(frames)} frames")

    cap.release()
    out.release()
    print(f"   - ✓ Visualization saved to: {out_path}")

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
    parser.add_argument('--nominal-depth', type=float, default=DEFAULT_NOMINAL_DEPTH_M, help='Nominal scene depth (meters) used to scale MiDaS outputs')
    parser.add_argument('--no-validate', action='store_true', help='Skip JSON schema validation after saving output')
    parser.add_argument('--viz-section', choices=['1','2','3','4','all'], default='3', help='Which visualization section to produce: 1,2,3,4 or all (default: 3)')
    args = parser.parse_args()

    # Decide whether to use CUDA: --gpu explicitly requests it, otherwise use_cuda unless --no-gpu set
    if args.gpu and not torch.cuda.is_available():
        raise SystemExit("Requested --gpu but CUDA is not available in this Python environment.")
    use_cuda = args.gpu or (not args.no_gpu)

    # Process frames (may be limited via CLI args)
    max_frames = args.max_frames if (hasattr(args, 'max_frames') and args.max_frames > 0) else total_frames
    print(f"   - Processing {max_frames} frames (sample_rate={args.sample_rate}, depth_downscale={args.depth_downscale}, nominal_depth={args.nominal_depth} m)")

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

    # Align IMU start to the first video timestamp so the 100Hz IMU stream
    # covers the 30fps video timeline (user requested IMU start → video start mapping).
    try:
        vid_start = float(timestamps[0])
        imu_start = float(imu_timestamps[0])
        shift = vid_start - imu_start
        if abs(shift) > 1e-6:
            imu_timestamps = imu_timestamps + shift
            print(f"   - Synchronized IMU start to video start (shifted IMU by {shift:.3f}s)")
    except Exception:
        pass

    print(f"   - Computed poses for {len(imu_timestamps)} IMU samples")
    # report IMU / video overlap so user knows where IMU-driven head motion will appear
    try:
        import numpy as _np
        imu_start, imu_end = float(imu_timestamps[0]), float(imu_timestamps[-1])
        vid_start, vid_end = float(timestamps[0]), float(timestamps[-1])
        overlap_start_idx = int(_np.searchsorted(timestamps, imu_start, side='left'))
        overlap_end_idx = int(_np.searchsorted(timestamps, imu_end, side='right') - 1)
        overlap_count = max(0, overlap_end_idx - overlap_start_idx + 1)
        print(f"   - IMU time coverage: {imu_start:.3f} — {imu_end:.3f} (video frames overlapped: {overlap_count})")
        if overlap_count == 0:
            print("     Warning: IMU samples do not overlap video timeline; head motion will not appear in most frames.")
    except Exception:
        pass
    print("   - Note: IMU available for fusion with visual odometry")
    
    print("\n4. Processing video frames...")
    # Pass IMU arrays so we can compute a loose VO+IMU fused orientation per frame
    frames = process_frames(cap, timestamps, hands, midas, midas_transforms, camera_matrix, max_frames, device=device, sample_rate=args.sample_rate, depth_downscale=args.depth_downscale, nominal_depth_m=args.nominal_depth, imu_poses=(imu_timestamps, orientation))
    
    print("\n5. Saving unified JSON...")
    save_unified_json(frames, fps)
    print("   - Saved to: output/unified_hand_tracking_android.json")

    # --- automatic JSON Schema validation (can be disabled with --no-validate) ---
    if not getattr(args, 'no_validate', False):
        valid = validate_unified_json()
        if not valid:
            raise SystemExit("Schema validation failed — aborting.")
    else:
        print("   - JSON schema validation skipped (--no-validate)")

    # --- short brief for available visualization sections ---
    print("\n6. Visualization options:")
    print("  1) Original video (raw frames)")
    print("  2) MediaPipe overlay (landmarks + connections)")
    print("  3) Minimal depth overlay — wrist depth only (default)")
    print("  4) 3D view — matplotlib 3D visualization (hides hands if head unstable)")
    print("  all) 2x2 composite showing sections 1–4\n")

    # Interactive selection when running from a terminal; otherwise use CLI value
    section = args.viz_section
    if sys.stdin.isatty():
        prompt = f"Select visualization to generate — enter 1, 2, 3, 4 or all [default: {args.viz_section}]: "
        while True:
            choice = input(prompt).strip().lower()
            if choice == "":
                choice = args.viz_section
            if choice in ("1", "2", "3", "4", "all"):
                section = choice
                break
            print("Invalid selection — please enter 1, 2, 3, 4 or all.")
    else:
        print(f"Non-interactive session — using --viz-section={args.viz_section}")

    print(f"Generating visualization: section={section}")
    generate_visualization(frames, section=section)

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()