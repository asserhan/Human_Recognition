from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import math
import time

# Configuration
VIDEO_PATH = "running_person.mp4"
OUTPUT_PATH = "output_running_detection.mp4"
MODEL_PATH = "yolov8n-pose.pt"

# BALANCED Detection parameters for CCTV
PRIMARY_VELOCITY_THRESHOLD = 200.0   # Main threshold (px/s)
SECONDARY_VELOCITY_THRESHOLD = 250.0 # Higher confidence threshold (px/s)
MIN_VELOCITY_THRESHOLD = 60.0        # Minimum to consider any movement
MAX_VELOCITY_THRESHOLD = 500.0       # Filter unrealistic speeds

MIN_FRAMES_FOR_ANALYSIS = 8       # Reduced for faster detection
CONFIDENCE_THRESHOLD = 0.5

# Enhanced filtering (optional features)
USE_VERTICAL_ANALYSIS = True         # Can be disabled for low-res CCTV
VERTICAL_MOVEMENT_THRESHOLD = 8.0    # Reduced threshold
USE_CONSISTENCY_CHECK = True         # Movement pattern consistency
CONSISTENCY_WINDOW = 8               # Frames to check for consistency

# Display settings
SHOW_KEYPOINTS = True
SHOW_TRAILS = True
TRAIL_LENGTH = 15

class BalancedPersonTracker:
    def __init__(self, track_id):
        self.track_id = track_id
        self.positions = deque(maxlen=30)
        self.keypoints_history = deque(maxlen=20)
        self.velocities = deque(maxlen=20)
        self.vertical_positions = deque(maxlen=25)
        
        # Current state
        self.current_velocity = 0.0
        self.vertical_movement = 0.0
        self.velocity_consistency = 0.0
        self.classification = "ANALYZING"
        self.confidence = 0.0
        
        # Frame counters
        self.total_frames = 0
        self.running_frames = 0
        self.not_running_frames = 0
        self.analyzing_frames = 0
        
        # Activity tracking
        self.activity_history = []
        self.first_frame = None
        self.last_frame = None
        
        # Detection state for stability
        self.consecutive_running = 0
        self.consecutive_not_running = 0
    
    def update(self, center_x, center_y, keypoints, frame_number, fps):
        """Update tracker with balanced analysis"""
        self.positions.append((center_x, center_y, frame_number))
        self.keypoints_history.append(keypoints)
        self.total_frames += 1
        
        # Track frame range
        if self.first_frame is None:
            self.first_frame = frame_number
        self.last_frame = frame_number
        
        # Extract position features
        self.extract_movement_features(keypoints)
        
        # Calculate primary metrics
        self.current_velocity = self.calculate_smoothed_velocity(fps)
        self.velocities.append(self.current_velocity)
        
        # Optional enhanced features
        if USE_VERTICAL_ANALYSIS:
            self.vertical_movement = self.calculate_vertical_movement()
        
        if USE_CONSISTENCY_CHECK:
            self.velocity_consistency = self.calculate_velocity_consistency()
        
        # BALANCED classification
        self.classify_movement_balanced()
        
        # Record activity
        self.activity_history.append({
            'frame': frame_number,
            'classification': self.classification,
            'velocity': self.current_velocity,
            'confidence': self.confidence,
            'vertical_movement': self.vertical_movement
        })
        
        # Update counters
        if self.classification == "RUNNING":
            self.running_frames += 1
            self.consecutive_running += 1
            self.consecutive_not_running = 0
        elif self.classification == "NOT RUNNING":
            self.not_running_frames += 1
            self.consecutive_not_running += 1
            self.consecutive_running = 0
        else:
            self.analyzing_frames += 1
            self.consecutive_running = 0
            self.consecutive_not_running = 0
    
    def extract_movement_features(self, keypoints):
        """Extract basic movement features"""
        if keypoints is None:
            return
        
        # Focus on hip center for main body tracking
        left_hip = self.get_keypoint_position(keypoints, 11)
        right_hip = self.get_keypoint_position(keypoints, 12)
        
        if left_hip and right_hip:
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            self.vertical_positions.append(hip_center_y)
        elif left_hip:
            self.vertical_positions.append(left_hip[1])
        elif right_hip:
            self.vertical_positions.append(right_hip[1])
    
    def get_keypoint_position(self, keypoints, keypoint_index):
        """Get position of specific keypoint"""
        if keypoint_index < len(keypoints):
            kpt = keypoints[keypoint_index]
            if len(kpt) >= 2:
                x, y = kpt[0], kpt[1]
                if not (np.isnan(x) or np.isnan(y)) and x > 0 and y > 0:
                    return (float(x), float(y))
        return None
    
    def calculate_smoothed_velocity(self, fps):
        """Calculate velocity with better smoothing and outlier removal"""
        if len(self.positions) < 3:
            return 0.0
        
        # Use recent positions for calculation
        recent_positions = list(self.positions)[-8:]
        
        if len(recent_positions) < 3:
            return 0.0
        
        # Calculate velocities between frames
        velocities = []
        for i in range(1, len(recent_positions)):
            x1, y1, f1 = recent_positions[i-1]
            x2, y2, f2 = recent_positions[i]
            
            dt = (f2 - f1) / fps
            if dt > 0:
                distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                velocity = distance / dt
                
                # Filter unrealistic velocities
                if MIN_VELOCITY_THRESHOLD * 0.5 <= velocity <= MAX_VELOCITY_THRESHOLD:
                    velocities.append(velocity)
        
        if not velocities:
            return 0.0
        
        # Use median to reduce noise impact
        median_velocity = np.median(velocities)
        
        # Light exponential smoothing if we have history
        if len(self.velocities) > 0:
            alpha = 0.4  # Moderate smoothing
            smoothed = alpha * median_velocity + (1 - alpha) * self.velocities[-1]
        else:
            smoothed = median_velocity
        
        return smoothed
    
    def calculate_vertical_movement(self):
        """Calculate vertical movement amplitude (simplified)"""
        if len(self.vertical_positions) < 8:
            return 0.0
        
        recent_y = list(self.vertical_positions)[-12:]
        if len(recent_y) < 5:
            return 0.0
        
        # Simple standard deviation of vertical position
        return np.std(recent_y)
    
    def calculate_velocity_consistency(self):
        """Check if velocity is reasonably consistent (not erratic)"""
        if len(self.velocities) < 6:
            return 0.0
        
        recent_velocities = list(self.velocities)[-CONSISTENCY_WINDOW:]
        
        if len(recent_velocities) < 4:
            return 0.0
        
        # Calculate how consistent velocities are
        mean_vel = np.mean(recent_velocities)
        if mean_vel <= 0:
            return 0.0
        
        std_vel = np.std(recent_velocities)
        cv = std_vel / mean_vel  # Coefficient of variation
        
        # Convert to consistency score (0-1, higher = more consistent)
        consistency = 1.0 / (1.0 + cv * 2)  # Reduce sensitivity
        return consistency
    
    def classify_movement_balanced(self):
        """BALANCED classification - prioritizes velocity but uses other features to filter"""
        if self.total_frames < MIN_FRAMES_FOR_ANALYSIS:
            self.classification = "ANALYZING"
            self.confidence = 0.0
            return
        
        # PRIMARY CRITERION: Velocity (most important for CCTV)
        avg_velocity = np.mean(list(self.velocities)[-6:]) if len(self.velocities) >= 3 else self.current_velocity
        
        # STRICT CLASSIFICATION based on velocity
        if avg_velocity >= PRIMARY_VELOCITY_THRESHOLD:
            # Only classify as RUNNING if velocity exceeds threshold
            base_classification = "RUNNING"
            if avg_velocity >= SECONDARY_VELOCITY_THRESHOLD:
                base_confidence = min(0.9, 0.7 + (avg_velocity - SECONDARY_VELOCITY_THRESHOLD) / 200.0)
            else:
                base_confidence = 0.6 + (avg_velocity - PRIMARY_VELOCITY_THRESHOLD) / (SECONDARY_VELOCITY_THRESHOLD - PRIMARY_VELOCITY_THRESHOLD) * 0.2
        else:
            base_classification = "NOT RUNNING"
            base_confidence = min(0.85, 0.6 + (PRIMARY_VELOCITY_THRESHOLD - avg_velocity) / PRIMARY_VELOCITY_THRESHOLD * 0.25)
        
        # OPTIONAL FEATURE ADJUSTMENTS (only modify, don't override)
        confidence_modifier = 1.0
        
        # Vertical movement check (if enabled and available)
        if USE_VERTICAL_ANALYSIS and len(self.vertical_positions) >= 8:
            if base_classification == "RUNNING":
                # For running classification, slight penalty if no vertical movement
                if self.vertical_movement < VERTICAL_MOVEMENT_THRESHOLD * 0.6:
                    confidence_modifier *= 0.85  # Small penalty
            else:
                # For not-running, boost confidence if there's little vertical movement
                if self.vertical_movement < VERTICAL_MOVEMENT_THRESHOLD * 0.3:
                    confidence_modifier *= 1.1
        
        # Consistency check (if enabled)
        if USE_CONSISTENCY_CHECK and len(self.velocities) >= 6:
            if base_classification == "RUNNING":
                # For running, require reasonable consistency
                if self.velocity_consistency < 0.3:  # Very erratic movement
                    confidence_modifier *= 0.8
                elif self.velocity_consistency > 0.6:  # Good consistency
                    confidence_modifier *= 1.05
        
        # Temporal stability check (prevent flickering)
        if len(self.activity_history) >= 5:
            recent_classifications = [a['classification'] for a in self.activity_history[-5:]]
            same_class_count = sum(1 for c in recent_classifications if c == base_classification)
            
            if same_class_count >= 3:  # Majority agreement
                confidence_modifier *= 1.1
            elif same_class_count <= 1:  # Minority - might be noise
                confidence_modifier *= 0.8
        
        # Apply modifiers
        final_confidence = min(0.95, max(0.2, base_confidence * confidence_modifier))
        
        # FINAL DECISION with stability
        if base_classification == "RUNNING" and final_confidence >= 0.45:
            self.classification = "RUNNING"
            self.confidence = final_confidence
        elif base_classification == "NOT RUNNING" and final_confidence >= 0.5:
            self.classification = "NOT RUNNING" 
            self.confidence = final_confidence
        else:
            # Low confidence - maintain previous state if available
            if hasattr(self, '_last_stable_classification') and self._last_stable_classification:
                self.classification = self._last_stable_classification
                self.confidence = max(0.3, final_confidence * 0.8)
            else:
                self.classification = "ANALYZING"
                self.confidence = final_confidence
        
        # Store stable classification for continuity
        if self.confidence >= 0.6:
            self._last_stable_classification = self.classification

    def get_overall_activity_summary(self, fps):
        """Get overall activity summary"""
        if not self.activity_history:
            return {
                'overall_classification': 'UNKNOWN',
                'confidence': 0.0,
                'running_percentage': 0.0,
                'total_time': 0.0,
                'running_time': 0.0,
                'not_running_time': 0.0,
                'analyzing_time': 0.0,
                'frames_tracked': 'N/A'
            }
        
        # Calculate metrics
        total_time = (self.last_frame - self.first_frame) / fps if self.first_frame and self.last_frame else 0.0
        running_time = self.running_frames / fps
        not_running_time = self.not_running_frames / fps
        analyzing_time = self.analyzing_frames / fps
        
        running_percentage = (self.running_frames / self.total_frames) * 100 if self.total_frames > 0 else 0
        
        # Determine overall classification
        trackable_frames = self.running_frames + self.not_running_frames
        if trackable_frames > 0:
            running_ratio = self.running_frames / trackable_frames
            
            # More reasonable thresholds for CCTV
            if running_ratio >= 0.25:  # 25% threshold - more sensitive
                overall_classification = "PREDOMINANTLY RUNNING"
                confidence = min(0.9, 0.5 + running_ratio * 0.4)
            elif running_ratio >= 0.08:  # 8% threshold - detect brief running
                overall_classification = "OCCASIONALLY RUNNING"
                confidence = min(0.8, 0.4 + running_ratio * 0.4)
            else:
                overall_classification = "MOSTLY STATIONARY/WALKING"
                confidence = min(0.85, 0.6 + (1 - running_ratio) * 0.25)
        else:
            overall_classification = "INSUFFICIENT DATA"
            confidence = 0.0
        
        return {
            'overall_classification': overall_classification,
            'confidence': confidence,
            'running_percentage': running_percentage,
            'total_time': total_time,
            'running_time': running_time,
            'not_running_time': not_running_time,
            'analyzing_time': analyzing_time,
            'max_velocity': max(self.velocities) if self.velocities else 0.0,
            'avg_velocity': np.mean(self.velocities) if self.velocities else 0.0,
            'avg_vertical_movement': np.mean([a['vertical_movement'] for a in self.activity_history]) if self.activity_history else 0.0,
            'frames_tracked': f"{self.first_frame}-{self.last_frame}" if self.first_frame and self.last_frame else "N/A"
        }

def calculate_center_of_mass(keypoints):
    """Calculate center of mass from keypoints"""
    if keypoints is None:
        return None
    
    # Priority: hips > shoulders > head
    priority_keypoints = [
        [11, 12],  # Left and right hip
        [5, 6],    # Left and right shoulder
        [0]        # Nose
    ]
    
    for keypoint_group in priority_keypoints:
        valid_points = []
        for kpt_idx in keypoint_group:
            pos = get_keypoint_position(keypoints, kpt_idx)
            if pos:
                valid_points.append(pos)
        
        if valid_points:
            avg_x = sum(p[0] for p in valid_points) / len(valid_points)
            avg_y = sum(p[1] for p in valid_points) / len(valid_points)
            return (avg_x, avg_y)
    
    return None

def get_keypoint_position(keypoints, keypoint_index):
    """Get position of specific keypoint"""
    if keypoint_index < len(keypoints):
        kpt = keypoints[keypoint_index]
        if len(kpt) >= 2:
            x, y = kpt[0], kpt[1]
            if not (np.isnan(x) or np.isnan(y)) and x > 0 and y > 0:
                return (float(x), float(y))
    return None

def draw_keypoints(frame, keypoints):
    """Draw important keypoints for running analysis"""
    if not SHOW_KEYPOINTS or keypoints is None:
        return
    
    # Key joints for running analysis
    key_joints = {
        11: (0, 255, 255),  # Left hip - cyan
        12: (0, 255, 255),  # Right hip - cyan
        13: (255, 255, 0),  # Left knee - yellow
        14: (255, 255, 0),  # Right knee - yellow
        15: (255, 0, 255),  # Left ankle - magenta
        16: (255, 0, 255),  # Right ankle - magenta
    }
    
    for joint_idx, color in key_joints.items():
        pos = get_keypoint_position(keypoints, joint_idx)
        if pos:
            x, y = pos
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)

def draw_balanced_person_info(frame, person, bbox):
    """Draw person information with balanced metrics"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Color and thickness based on classification
    if person.classification == "RUNNING":
        color = (0, 0, 255)  # Red
        thickness = 3
    elif person.classification == "NOT RUNNING":
        color = (0, 255, 0)  # Green
        thickness = 2
    else:  # ANALYZING
        color = (0, 255, 255)  # Yellow
        thickness = 2
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Information text
    status_text = f"ID:{person.track_id} - {person.classification}"
    velocity_text = f"Velocity: {person.current_velocity:.1f} px/s"
    
    # Additional metrics (if enabled)
    metrics_text = ""
    if USE_VERTICAL_ANALYSIS:
        metrics_text += f" | V.Move: {person.vertical_movement:.1f}"
    if USE_CONSISTENCY_CHECK:
        metrics_text += f" | Consist: {person.velocity_consistency:.2f}"
    
    confidence_text = f"Confidence: {person.confidence:.2f}{metrics_text}"
    
    # Draw text background
    text_y = y1 - 15
    cv2.rectangle(frame, (x1, text_y - 65), (x1 + 450, text_y + 10), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, status_text, (x1 + 5, text_y - 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, velocity_text, (x1 + 5, text_y - 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, confidence_text, (x1 + 5, text_y - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw movement trail
    if SHOW_TRAILS and len(person.positions) > 1:
        positions = list(person.positions)[-TRAIL_LENGTH:]
        for i in range(1, len(positions)):
            x1_trail, y1_trail, _ = positions[i-1]
            x2_trail, y2_trail, _ = positions[i]
            
            alpha = i / len(positions)
            trail_color = tuple(int(c * alpha) for c in color)
            
            cv2.line(frame, (int(x1_trail), int(y1_trail)), 
                    (int(x2_trail), int(y2_trail)), trail_color, 2)

def draw_balanced_statistics(frame, persons, frame_count, total_frames, fps):
    """Draw system statistics with balanced approach info"""
    height, width = frame.shape[:2]
    
    # Background
    cv2.rectangle(frame, (0, 0), (width, 120), (0, 0, 0), -1)
    
    # Count people by status
    running_count = sum(1 for p in persons.values() if p.classification == "RUNNING")
    not_running_count = sum(1 for p in persons.values() if p.classification == "NOT RUNNING")
    analyzing_count = sum(1 for p in persons.values() if p.classification == "ANALYZING")
    
    # Progress
    progress_percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
    time_in_video = frame_count / fps
    
    # Statistics display
    stats_text = f"🔴 RUNNING: {running_count} | 🟢 NOT RUNNING: {not_running_count} | 🟡 ANALYZING: {analyzing_count}"
    cv2.putText(frame, stats_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    progress_text = f"Progress: {progress_percent:.1f}% | Time: {time_in_video:.1f}s | Frame: {frame_count}/{total_frames}"
    cv2.putText(frame, progress_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    config_text = f"Balanced Mode | Primary: {PRIMARY_VELOCITY_THRESHOLD:.0f} px/s | Secondary: {SECONDARY_VELOCITY_THRESHOLD:.0f} px/s"
    cv2.putText(frame, config_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    features_text = f"Features: Velocity"
    if USE_VERTICAL_ANALYSIS:
        features_text += " + Vertical"
    if USE_CONSISTENCY_CHECK:
        features_text += " + Consistency"
    features_text += " | Press 'q' to quit"
    cv2.putText(frame, features_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

def main():
    """Main function with balanced detection approach"""
    # Check if video exists
    import os
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found: {VIDEO_PATH}")
        return
    
    # Initialize YOLO model
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {VIDEO_PATH}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    # Person trackers
    persons = {}
    frame_count = 0
    
    # Statistics
    total_running_detections = 0
    high_confidence_alerts = 0
    medium_confidence_alerts = 0
    
    print("🎯 BALANCED RUNNING DETECTION FOR CCTV")
    print("📋 Strategy: Velocity-primary with optional feature enhancement")
    print(f"⚙️  Primary Velocity Threshold: {PRIMARY_VELOCITY_THRESHOLD} px/s")
    print(f"⚙️  Secondary Velocity Threshold: {SECONDARY_VELOCITY_THRESHOLD} px/s")
    print(f"⚙️  Vertical Analysis: {'Enabled' if USE_VERTICAL_ANALYSIS else 'Disabled'}")
    print(f"⚙️  Consistency Check: {'Enabled' if USE_CONSISTENCY_CHECK else 'Disabled'}")
    print("Press 'q' to quit early")
    print("-" * 80)

    start_time = time.time()
    
    try:
        # Process video
        for result in model.track(source=VIDEO_PATH, stream=True, persist=True, verbose=False, conf=CONFIDENCE_THRESHOLD):
            frame = result.orig_img.copy()
            frame_count += 1
            
            # Get detections
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                keypoints_all = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
                
                # Process each detected person
                for i, track_id in enumerate(track_ids):
                    bbox = boxes[i]
                    keypoints = keypoints_all[i] if keypoints_all is not None else None
                    
                    # Calculate center position
                    x1, y1, x2, y2 = bbox
                    
                    # Use center of mass from keypoints or bbox center
                    center_pos = calculate_center_of_mass(keypoints) if keypoints is not None else None
                    if center_pos:
                        center_x, center_y = center_pos
                    else:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                    
                    # Initialize or update tracker
                    if track_id not in persons:
                        persons[track_id] = BalancedPersonTracker(track_id)
                    
                    # Update tracker
                    persons[track_id].update(center_x, center_y, keypoints, frame_count, fps)
                    
                    # BALANCED alert system
                    person = persons[track_id]
                    if person.classification == "RUNNING":
                        total_running_detections += 1
                        
                        # High confidence alerts (every 30 frames = 1 second)
                        if person.confidence >= 0.7 and frame_count % 30 == 0:
                            time_in_video = frame_count / fps
                            print(f"🏃 HIGH CONFIDENCE RUNNING! Person {track_id} at {time_in_video:.1f}s")
                            print(f"   └─ Velocity: {person.current_velocity:.1f} px/s | Confidence: {person.confidence:.2f}")
                            high_confidence_alerts += 1
                        
                        # Medium confidence alerts (every 60 frames = 2 seconds)
                        elif person.confidence >= 0.5 and frame_count % 60 == 0:
                            time_in_video = frame_count / fps
                            print(f"🤔 MEDIUM CONFIDENCE RUNNING: Person {track_id} at {time_in_video:.1f}s (Conf: {person.confidence:.2f})")
                            medium_confidence_alerts += 1
                    
                    # Draw information
                    draw_balanced_person_info(frame, person, bbox)
                    
                    # Draw keypoints
                    if keypoints is not None:
                        draw_keypoints(frame, keypoints)
            
            # Clean up old persons
            current_ids = set()
            if result.boxes is not None and result.boxes.id is not None:
                current_ids = set(result.boxes.id.cpu().numpy().astype(int))
            
            persons_to_remove = []
            for track_id, person in persons.items():
                if track_id not in current_ids:
                    if len(person.positions) > 0:
                        last_seen_frame = person.positions[-1][2]
                        if frame_count - last_seen_frame > 75:  # 2.5 seconds
                            persons_to_remove.append(track_id)
            
            for track_id in persons_to_remove:
                del persons[track_id]
            
            # Draw statistics
            draw_balanced_statistics(frame, persons, frame_count, total_frames, fps)
            
            # Write and display frame
            out.write(frame)
            cv2.imshow('Balanced Running Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Progress update
            if frame_count % 60 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed_time = time.time() - start_time
                processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                running_velocities = [(track_id, person.current_velocity) for track_id, person in persons.items() if person.classification == "RUNNING"]
                running_str = ', '.join([f"ID:{tid}={vel:.1f}px/s" for tid, vel in running_velocities]) if running_velocities else 'None'
                print(f"📊 Progress: {progress:.1f}% | Processing: {processing_fps:.1f} FPS | Total Running: {total_running_detections} | High-Conf: {high_confidence_alerts} | Running Velocities: [{running_str}]")

    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Final comprehensive report
        processing_time = time.time() - start_time
        video_duration = total_frames / fps
        
        print(f"\n" + "="*100)
        print(f"🎬 BALANCED RUNNING DETECTION ANALYSIS COMPLETE")
        print(f"="*100)
        print(f"📁 Input: {VIDEO_PATH}")
        print(f"📁 Output: {OUTPUT_PATH}")
        print(f"⏱️  Video Duration: {video_duration:.1f}s | Processing Time: {processing_time:.1f}s")
        print(f"📊 Frames Processed: {frame_count}/{total_frames}")
        print(f"👥 Persons Detected: {len(persons)}")
        print(f"🏃 Total Running Detections: {total_running_detections}")
        print(f"🎯 High-Confidence Alerts: {high_confidence_alerts}")
        print(f"🤔 Medium-Confidence Alerts: {medium_confidence_alerts}")
        
        # Individual person summaries
        if persons:
            print(f"\n📋 INDIVIDUAL PERSON ANALYSIS:")
            print(f"-" * 100)
            for track_id, person in persons.items():
                summary = person.get_overall_activity_summary(fps)
                print(f"👤 Person {track_id}:")
                print(f"   🎯 Overall: {summary['overall_classification']} (Confidence: {summary['confidence']:.2f})")
                print(f"   📊 Running Time: {summary['running_time']:.1f}s ({summary['running_percentage']:.1f}%)")
                print(f"   📊 Not Running Time: {summary['not_running_time']:.1f}s")
                print(f"   📊 Max Velocity: {summary['max_velocity']:.1f} px/s | Avg: {summary['avg_velocity']:.1f} px/s")
                if USE_VERTICAL_ANALYSIS and summary['avg_vertical_movement'] > 0:
                    print(f"   📊 Avg Vertical Movement: {summary['avg_vertical_movement']:.1f}")
                print(f"   📊 Frames Tracked: {summary['frames_tracked']}")
                print()
        
        print(f"✅ Balanced analysis complete! Check {OUTPUT_PATH} for annotated video.")
        print(f"\n💡 TUNING TIPS:")
        print(f"   - If missing running: Lower PRIMARY_VELOCITY_THRESHOLD (currently {PRIMARY_VELOCITY_THRESHOLD})")
        print(f"   - If too many false positives: Raise PRIMARY_VELOCITY_THRESHOLD or enable more features")
        print(f"   - For low-res CCTV: Set USE_VERTICAL_ANALYSIS = False")
        print(f"   - For high-res videos: Set USE_VERTICAL_ANALYSIS = True, lower VERTICAL_MOVEMENT_THRESHOLD")

if __name__ == "__main__":
    main()