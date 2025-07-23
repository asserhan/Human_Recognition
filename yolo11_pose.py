from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import imageio
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

# ---------------- CONFIG ----------------
video_path = "running_person.mp4"
output_path = "output_running.mp4"
model_path = "yolov8n-pose.pt"

# Analysis parameters
ANALYSIS_WINDOW = 30          # Frames to analyze per person
MIN_DETECTION_FRAMES = 15     # Minimum frames needed for classification
CONFIDENCE_THRESHOLD = 0.75   # Final classification confidence threshold

# Biomechanical thresholds (research-based)
STRIDE_FREQUENCY_THRESHOLD = 2.5    # Steps per second (running > 2.5)
FLIGHT_TIME_THRESHOLD = 0.1         # Seconds both feet off ground
VERTICAL_OSCILLATION_THRESHOLD = 0.15  # Body bounce ratio
CADENCE_THRESHOLD = 160             # Steps per minute (running > 160)
GROUND_CONTACT_RATIO_THRESHOLD = 0.4  # Running < 0.4, walking > 0.6

# ----------------------------------------

@dataclass
class GaitMetrics:
    """Store biomechanical gait analysis metrics"""
    stride_length: float = 0.0
    stride_frequency: float = 0.0
    vertical_oscillation: float = 0.0
    ground_contact_time: float = 0.0
    flight_time: float = 0.0
    cadence: float = 0.0
    knee_drive: float = 0.0
    arm_swing_amplitude: float = 0.0

@dataclass
class PersonTracker:
    track_id: int
    keypoint_history: deque  # Store complete keypoint data
    position_history: deque  # Center of mass positions
    gait_metrics_history: deque
    frame_timestamps: deque
    
    # Analysis results
    final_classification: str = "ANALYZING"
    confidence: float = 0.0
    analysis_complete: bool = False
    
    # Biomechanical state
    last_foot_contact: Dict = None
    stride_count: int = 0
    current_gait_cycle: List = None
    
    def __post_init__(self):
        if self.last_foot_contact is None:
            self.last_foot_contact = {"left": None, "right": None}
        if self.current_gait_cycle is None:
            self.current_gait_cycle = []

class RunningDetectionAlgorithm:
    """Advanced running detection based on biomechanical analysis"""
    
    def __init__(self, fps: float):
        self.fps = fps
        self.keypoint_names = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
    
    def get_valid_keypoint(self, keypoints: np.ndarray, name: str) -> Optional[Tuple[float, float]]:
        """Safely extract keypoint coordinates"""
        try:
            idx = self.keypoint_names[name]
            if idx < len(keypoints):
                x, y = keypoints[idx]
                if not (np.isnan(x) or np.isnan(y)) and x > 0 and y > 0:
                    return (float(x), float(y))
        except (IndexError, KeyError):
            pass
        return None
    
    def calculate_center_of_mass(self, keypoints: np.ndarray) -> Optional[Tuple[float, float]]:
        """Calculate center of mass from available keypoints"""
        # Priority: hips > shoulders > torso average
        left_hip = self.get_valid_keypoint(keypoints, 'left_hip')
        right_hip = self.get_valid_keypoint(keypoints, 'right_hip')
        
        if left_hip and right_hip:
            return ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
        
        left_shoulder = self.get_valid_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self.get_valid_keypoint(keypoints, 'right_shoulder')
        
        if left_shoulder and right_shoulder:
            return ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
        
        # Single point fallback
        for point_name in ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']:
            point = self.get_valid_keypoint(keypoints, point_name)
            if point:
                return point
        
        return None
    
    def detect_foot_contact(self, keypoints: np.ndarray, ground_level: float) -> Dict[str, bool]:
        """Detect if feet are in contact with ground"""
        contacts = {"left": False, "right": False}
        
        for side in ['left', 'right']:
            ankle = self.get_valid_keypoint(keypoints, f'{side}_ankle')
            if ankle:
                # Consider foot in contact if ankle is close to ground level
                contacts[side] = abs(ankle[1] - ground_level) < 20  # 20 pixel threshold
        
        return contacts
    
    def calculate_stride_metrics(self, person: PersonTracker) -> GaitMetrics:
        """Calculate comprehensive gait metrics"""
        if len(person.keypoint_history) < 10:
            return GaitMetrics()
        
        metrics = GaitMetrics()
        keypoint_data = list(person.keypoint_history)
        position_data = list(person.position_history)
        
        # 1. Vertical oscillation (body bounce)
        if len(position_data) >= 10:
            y_positions = [pos[1] for pos in position_data]
            y_range = max(y_positions) - min(y_positions)
            body_height = self.estimate_body_height(keypoint_data[-1])
            if body_height > 0:
                metrics.vertical_oscillation = y_range / body_height
        
        # 2. Stride frequency analysis
        metrics.stride_frequency = self.calculate_stride_frequency(person)
        
        # 3. Ground contact analysis
        ground_level = self.estimate_ground_level(keypoint_data)
        contact_ratios = self.analyze_ground_contact(keypoint_data, ground_level)
        metrics.ground_contact_time = contact_ratios.get('avg_contact_ratio', 0.5)
        
        # 4. Knee drive analysis (knee lift height)
        metrics.knee_drive = self.calculate_knee_drive(keypoint_data)
        
        # 5. Arm swing amplitude
        metrics.arm_swing_amplitude = self.calculate_arm_swing(keypoint_data)
        
        # 6. Cadence (steps per minute)
        if metrics.stride_frequency > 0:
            metrics.cadence = metrics.stride_frequency * 60 * 2  # 2 steps per stride
        
        return metrics
    
    def estimate_body_height(self, keypoints: np.ndarray) -> float:
        """Estimate person's height from keypoints"""
        # Try head to feet
        head_points = ['nose', 'left_eye', 'right_eye']
        foot_points = ['left_ankle', 'right_ankle']
        
        head_y = None
        for point_name in head_points:
            point = self.get_valid_keypoint(keypoints, point_name)
            if point:
                head_y = point[1]
                break
        
        foot_y = None
        for point_name in foot_points:
            point = self.get_valid_keypoint(keypoints, point_name)
            if point:
                foot_y = max(foot_y or 0, point[1])  # Lowest foot
        
        if head_y and foot_y and foot_y > head_y:
            return foot_y - head_y
        
        # Fallback: hip to shoulder distance * 3.5 (rough body proportion)
        left_hip = self.get_valid_keypoint(keypoints, 'left_hip')
        left_shoulder = self.get_valid_keypoint(keypoints, 'left_shoulder')
        
        if left_hip and left_shoulder:
            torso_height = abs(left_hip[1] - left_shoulder[1])
            return torso_height * 3.5
        
        return 100.0  # Default fallback
    
    def estimate_ground_level(self, keypoint_history: List) -> float:
        """Estimate ground level from ankle positions"""
        all_ankle_y = []
        
        for keypoints in keypoint_history:
            for side in ['left', 'right']:
                ankle = self.get_valid_keypoint(keypoints, f'{side}_ankle')
                if ankle:
                    all_ankle_y.append(ankle[1])
        
        if all_ankle_y:
            # Use 90th percentile as ground level (highest ankle positions)
            return np.percentile(all_ankle_y, 90)
        
        return 480  # Default for 640x480, adjust based on video
    
    def calculate_stride_frequency(self, person: PersonTracker) -> float:
        """Calculate stride frequency from foot contact patterns"""
        if len(person.keypoint_history) < 20:
            return 0.0
        
        keypoint_data = list(person.keypoint_history)
        ground_level = self.estimate_ground_level(keypoint_data)
        
        # Detect foot strikes
        left_strikes = []
        right_strikes = []
        
        for i, keypoints in enumerate(keypoint_data[1:], 1):
            prev_keypoints = keypoint_data[i-1]
            
            # Detect left foot strike
            left_ankle = self.get_valid_keypoint(keypoints, 'left_ankle')
            prev_left_ankle = self.get_valid_keypoint(prev_keypoints, 'left_ankle')
            
            if left_ankle and prev_left_ankle:
                # Strike detected: foot moves down and gets close to ground
                if (left_ankle[1] > prev_left_ankle[1] and 
                    abs(left_ankle[1] - ground_level) < 15):
                    left_strikes.append(i)
            
            # Detect right foot strike
            right_ankle = self.get_valid_keypoint(keypoints, 'right_ankle')
            prev_right_ankle = self.get_valid_keypoint(prev_keypoints, 'right_ankle')
            
            if right_ankle and prev_right_ankle:
                if (right_ankle[1] > prev_right_ankle[1] and 
                    abs(right_ankle[1] - ground_level) < 15):
                    right_strikes.append(i)
        
        # Calculate frequency from strike intervals
        all_strikes = sorted(left_strikes + right_strikes)
        if len(all_strikes) >= 4:
            # Calculate average time between strikes
            intervals = [all_strikes[i+1] - all_strikes[i] for i in range(len(all_strikes)-1)]
            avg_interval_frames = np.mean(intervals)
            avg_interval_seconds = avg_interval_frames / self.fps
            
            # Frequency = 1 / (2 * average_strike_interval) for stride frequency
            return 1.0 / (2.0 * avg_interval_seconds) if avg_interval_seconds > 0 else 0.0
        
        return 0.0
    
    def analyze_ground_contact(self, keypoint_history: List, ground_level: float) -> Dict:
        """Analyze ground contact patterns"""
        contact_ratios = {"left": [], "right": []}
        
        for keypoints in keypoint_history:
            for side in ['left', 'right']:
                ankle = self.get_valid_keypoint(keypoints, f'{side}_ankle')
                if ankle:
                    is_contact = abs(ankle[1] - ground_level) < 20
                    contact_ratios[side].append(1.0 if is_contact else 0.0)
        
        result = {}
        for side in ['left', 'right']:
            if contact_ratios[side]:
                result[f'{side}_contact_ratio'] = np.mean(contact_ratios[side])
        
        # Average contact ratio
        if result:
            result['avg_contact_ratio'] = np.mean(list(result.values()))
        else:
            result['avg_contact_ratio'] = 0.5
        
        return result
    
    def calculate_knee_drive(self, keypoint_history: List) -> float:
        """Calculate maximum knee lift (knee drive)"""
        max_knee_lifts = []
        
        for keypoints in keypoint_history:
            for side in ['left', 'right']:
                hip = self.get_valid_keypoint(keypoints, f'{side}_hip')
                knee = self.get_valid_keypoint(keypoints, f'{side}_knee')
                
                if hip and knee:
                    # Knee lift = how high knee goes relative to hip
                    knee_lift = hip[1] - knee[1]  # Positive = knee above hip
                    max_knee_lifts.append(max(0, knee_lift))
        
        return max(max_knee_lifts) if max_knee_lifts else 0.0
    
    def calculate_arm_swing(self, keypoint_history: List) -> float:
        """Calculate arm swing amplitude"""
        arm_positions = {"left": [], "right": []}
        
        for keypoints in keypoint_history:
            for side in ['left', 'right']:
                wrist = self.get_valid_keypoint(keypoints, f'{side}_wrist')
                shoulder = self.get_valid_keypoint(keypoints, f'{side}_shoulder')
                
                if wrist and shoulder:
                    # Arm angle relative to vertical
                    dx = wrist[0] - shoulder[0]
                    dy = wrist[1] - shoulder[1]
                    angle = math.atan2(dx, dy)
                    arm_positions[side].append(angle)
        
        total_swing = 0.0
        for side in ['left', 'right']:
            if len(arm_positions[side]) > 5:
                angles = arm_positions[side]
                swing_range = max(angles) - min(angles)
                total_swing += swing_range
        
        return total_swing
    
    def classify_running(self, metrics: GaitMetrics) -> Tuple[str, float]:
        """
        Classify running vs not running based on biomechanical metrics
        Research-based classification algorithm
        """
        running_indicators = 0
        total_indicators = 0
        confidence_factors = []
        
        # 1. Stride frequency (most reliable indicator)
        if metrics.stride_frequency > 0:
            total_indicators += 1
            if metrics.stride_frequency > STRIDE_FREQUENCY_THRESHOLD:
                running_indicators += 1
                confidence_factors.append(min(metrics.stride_frequency / 3.5, 1.0))
            else:
                confidence_factors.append(0.2)
        
        # 2. Vertical oscillation (body bounce)
        if metrics.vertical_oscillation > 0:
            total_indicators += 1
            if metrics.vertical_oscillation > VERTICAL_OSCILLATION_THRESHOLD:
                running_indicators += 1
                confidence_factors.append(min(metrics.vertical_oscillation / 0.3, 1.0))
            else:
                confidence_factors.append(0.3)
        
        # 3. Ground contact ratio (lower for running)
        total_indicators += 1
        if metrics.ground_contact_time < GROUND_CONTACT_RATIO_THRESHOLD:
            running_indicators += 1
            confidence_factors.append(1.0 - metrics.ground_contact_time)
        else:
            confidence_factors.append(0.4)
        
        # 4. Cadence (steps per minute)
        if metrics.cadence > 0:
            total_indicators += 1
            if metrics.cadence > CADENCE_THRESHOLD:
                running_indicators += 1
                confidence_factors.append(min(metrics.cadence / 200, 1.0))
            else:
                confidence_factors.append(0.3)
        
        # 5. Knee drive (higher for running)
        if metrics.knee_drive > 0:
            total_indicators += 1
            if metrics.knee_drive > 30:  # 30 pixels threshold
                running_indicators += 1
                confidence_factors.append(min(metrics.knee_drive / 60, 1.0))
            else:
                confidence_factors.append(0.2)
        
        # 6. Arm swing amplitude
        if metrics.arm_swing_amplitude > 0:
            total_indicators += 1
            if metrics.arm_swing_amplitude > 1.0:  # Radians
                running_indicators += 1
                confidence_factors.append(min(metrics.arm_swing_amplitude / 2.0, 1.0))
            else:
                confidence_factors.append(0.3)
        
        # Final classification
        if total_indicators == 0:
            return "NOT RUNNING", 0.0
        
        running_ratio = running_indicators / total_indicators
        confidence = np.mean(confidence_factors) if confidence_factors else 0.0
        
        # Classification thresholds
        if running_ratio >= 0.5 and confidence >= CONFIDENCE_THRESHOLD:
            return "RUNNING", confidence
        elif running_ratio >= 0.4 and confidence >= 0.6:
            return "RUNNING", confidence * 0.8  # Lower confidence
        else:
            return "NOT RUNNING", 1.0 - confidence

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

print(f"Input: {width}x{height} @ {fps:.1f} FPS ({total_frames} frames)")
print(f"Analysis window: {ANALYSIS_WINDOW} frames per person")
print(f"Minimum detection frames: {MIN_DETECTION_FRAMES}")
print(f"Output will be saved as: {output_path}")

# Initialize algorithm and tracking
algorithm = RunningDetectionAlgorithm(fps)
writer = imageio.get_writer(output_path, fps=fps)
persons: Dict[int, PersonTracker] = {}
frame_idx = 0

# Stream YOLO detections
for r in model.track(source=video_path, stream=True, persist=True, verbose=False):
    frame = r.orig_img
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
    ids = r.boxes.id.cpu().numpy().astype(int) if (r.boxes and r.boxes.id is not None) else []
    kpts_all = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else []
    
    for i, track_id in enumerate(ids):
        x1, y1, x2, y2 = boxes[i]
        keypoints = kpts_all[i] if len(kpts_all) > i else np.full((17, 2), np.nan)

        # Initialize person tracking if new
        if track_id not in persons:
            persons[track_id] = PersonTracker(
                track_id=track_id,
                keypoint_history=deque(maxlen=ANALYSIS_WINDOW),
                position_history=deque(maxlen=ANALYSIS_WINDOW),
                gait_metrics_history=deque(maxlen=10),
                frame_timestamps=deque(maxlen=ANALYSIS_WINDOW),
            )

        person = persons[track_id]
        
        # Store keypoint data and position
        person.keypoint_history.append(keypoints)
        person.frame_timestamps.append(frame_idx)
        
        center_of_mass = algorithm.calculate_center_of_mass(keypoints)
        if center_of_mass:
            person.position_history.append(center_of_mass)
        
        # Perform analysis if enough data collected
        if (len(person.keypoint_history) >= MIN_DETECTION_FRAMES and 
            not person.analysis_complete):
            
            # Calculate gait metrics
            metrics = algorithm.calculate_stride_metrics(person)
            person.gait_metrics_history.append(metrics)
            
            # Classify if we have enough analysis
            if len(person.gait_metrics_history) >= 3:
                # Use latest metrics for classification
                latest_metrics = person.gait_metrics_history[-1]
                classification, confidence = algorithm.classify_running(latest_metrics)
                
                person.final_classification = classification
                person.confidence = confidence
                person.analysis_complete = True
        
        # Choose color based on classification
        if person.analysis_complete:
            color = (0, 0, 255) if person.final_classification == "RUNNING" else (0, 255, 0)  # Red/Green
            status_text = f"ID:{track_id} - {person.final_classification}"
        else:
            color = (255, 255, 0)  # Yellow for analyzing
            status_text = f"ID:{track_id} - ANALYZING ({len(person.keypoint_history)}/{MIN_DETECTION_FRAMES})"
        
        # Draw bounding box
        thickness = 3 if person.final_classification == "RUNNING" else 2
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        # Draw status and metrics
        cv2.putText(frame, status_text, (int(x1), int(y1) - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if person.analysis_complete:
            confidence_text = f"Confidence: {person.confidence:.2f}"
            cv2.putText(frame, confidence_text, (int(x1), int(y1) - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Show key metrics
            if person.gait_metrics_history:
                latest = person.gait_metrics_history[-1]
                metrics_text = f"Freq:{latest.stride_frequency:.1f} Osc:{latest.vertical_oscillation:.2f}"
                cv2.putText(frame, metrics_text, (int(x1), int(y1) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                cadence_text = f"Cadence:{latest.cadence:.0f} Contact:{latest.ground_contact_time:.2f}"
                cv2.putText(frame, cadence_text, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw keypoints
        for idx, kpt in enumerate(keypoints):
            if len(kpt) >= 2 and not (np.isnan(kpt[0]) or np.isnan(kpt[1])):
                x_kpt, y_kpt = int(kpt[0]), int(kpt[1])
                # Highlight important keypoints for gait analysis
                if idx in [11, 12, 13, 14, 15, 16]:  # Legs and hips
                    cv2.circle(frame, (x_kpt, y_kpt), 4, (255, 255, 255), -1)
                elif idx in [5, 6, 9, 10]:  # Arms
                    cv2.circle(frame, (x_kpt, y_kpt), 3, (0, 255, 255), -1)
                else:
                    cv2.circle(frame, (x_kpt, y_kpt), 2, (100, 100, 100), -1)
        
        # Highlight center of mass
        if center_of_mass:
            cv2.circle(frame, (int(center_of_mass[0]), int(center_of_mass[1])), 6, (0, 255, 255), -1)

    # Draw overall statistics
    cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)
    
    # Count classifications
    running_count = sum(1 for p in persons.values() 
                       if p.analysis_complete and p.final_classification == "RUNNING")
    not_running_count = sum(1 for p in persons.values() 
                           if p.analysis_complete and p.final_classification == "NOT RUNNING")
    analyzing_count = sum(1 for p in persons.values() if not p.analysis_complete)
    
    stats_text = f"RUNNING: {running_count} | NOT RUNNING: {not_running_count} | ANALYZING: {analyzing_count}"
    cv2.putText(frame, stats_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    frame_text = f"Frame: {frame_idx}/{total_frames}"
    cv2.putText(frame, frame_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Analysis parameters
    params_text = f"Analysis: {ANALYSIS_WINDOW}f window, {MIN_DETECTION_FRAMES}f min, {CONFIDENCE_THRESHOLD:.1f} conf"
    cv2.putText(frame, params_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Write frame
    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_idx += 1
    
    if frame_idx % 60 == 0:
        print(f"Processed {frame_idx}/{total_frames} frames - R:{running_count} NR:{not_running_count} A:{analyzing_count}")

writer.close()

# Final report
print(f"\n✅ Analysis Complete! Output saved to: {output_path}")
print(f"\n📊 Final Results:")
for track_id, person in persons.items():
    if person.analysis_complete:
        print(f"Person {track_id}: {person.final_classification} (Confidence: {person.confidence:.2f})")
        if person.gait_metrics_history:
            metrics = person.gait_metrics_history[-1]
            print(f"  - Stride Frequency: {metrics.stride_frequency:.2f} Hz")
            print(f"  - Vertical Oscillation: {metrics.vertical_oscillation:.3f}")
            print(f"  - Ground Contact Ratio: {metrics.ground_contact_time:.3f}")
            print(f"  - Cadence: {metrics.cadence:.0f} steps/min")
    else:
        print(f"Person {track_id}: Insufficient data for analysis")

print(f"\nTotal persons analyzed: {len([p for p in persons.values() if p.analysis_complete])}")
print(f"Running: {sum(1 for p in persons.values() if p.final_classification == 'RUNNING')}")
print(f"Not Running: {sum(1 for p in persons.values() if p.final_classification == 'NOT RUNNING')}")