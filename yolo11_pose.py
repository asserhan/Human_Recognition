from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import imageio
import os

# ---------------- CONFIG ----------------
video_path = "running_person.mp4"  # Input video
output_path = "output_running.mp4"  # Output video
model_path = "yolov8n-pose.pt"     # YOLO pose model
RUN_THRESHOLD = 0.8                # Speed threshold
WINDOW = 5                         # Frame history for smoothing
# ----------------------------------------

# Check video exists
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found: {video_path}")

# Load model
model = YOLO(model_path)

# Get video metadata
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(f"Input: {width}x{height} @ {fps} FPS ({total_frames} frames)")
print(f"Output will be saved as: {output_path}")

# ImageIO writer (FFmpeg backend)
writer = imageio.get_writer(output_path, fps=fps)

# Tracking state
history = {}
frame_idx = 0

# Stream YOLO detections
for r in model.track(source=video_path, stream=True, persist=True, verbose=False):
    frame = r.orig_img  # original BGR frame
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
    ids = r.boxes.id.cpu().numpy().astype(int) if (r.boxes and r.boxes.id is not None) else []
    kpts_all = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else []

    for i, track_id in enumerate(ids):
        x1, y1, x2, y2 = boxes[i]
        kpts = kpts_all[i] if len(kpts_all) > i else np.full((17, 2), np.nan)

        # Anchor point: hips midpoint else bbox center
        if not np.isnan(kpts[11]).any() and not np.isnan(kpts[12]).any():
            cx = (kpts[11, 0] + kpts[12, 0]) / 2
            cy = (kpts[11, 1] + kpts[12, 1]) / 2
        else:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

        size = max(y2 - y1, 1e-6)
        dq = history.setdefault(track_id, deque(maxlen=WINDOW))
        dq.append((frame_idx, cx, cy, size))

        # Compute relative speed
        rel_speed = 0
        if len(dq) >= 2:
            speeds = []
            prev = dq[0]
            for curr in list(dq)[1:]:
                f0, x0, y0, s0 = prev
                f1, x1_, y1_, s1 = curr
                dt = (f1 - f0) / fps
                if dt > 0:
                    dist = np.hypot(x1_ - x0, y1_ - y0)
                    scale = (s0 + s1) / 2
                    speeds.append((dist / scale) / dt)
                prev = curr
            if speeds:
                rel_speed = np.median(speeds)

        # Determine status
        status = "RUNNING" if rel_speed > RUN_THRESHOLD else "NOT RUNNING"
        color = (0, 0, 255) if status == "RUNNING" else (0, 255, 0)

        # Draw box + labels
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{status}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Speed: {rel_speed:.2f}", (int(x1), int(y1) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw keypoints
        for (kx, ky) in kpts:
            if not np.isnan(kx) and not np.isnan(ky):
                cv2.circle(frame, (int(kx), int(ky)), 2, (255, 255, 255), -1)

    # Write frame (convert BGR → RGB for imageio)
    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"Processed {frame_idx}/{total_frames} frames")

writer.close()
print(f"\n✅ Done! Output saved to: {output_path}")
