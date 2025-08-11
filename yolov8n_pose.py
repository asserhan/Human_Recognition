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

# Detection parameters
VELOCITY_THRESHOLD = 120.0  # pixels/second - adjust based on your video
MIN_FRAMES_FOR_ANALYSIS = 10
CONFIDENCE_THRESHOLD = 0.5

# Display settings
SHOW_KEYPOINTS = True
SHOW_TRAILS = True
TRAIL_LENGTH = 20


class SimplePersonTracker:
    def __init__(self, track_id):
        self.track_id = track_id
        self.positions = deque(maxlen=30)  # (x, y, frame_number)
        self.keypoints_history = deque(maxlen=20)
        self.velocities = deque(maxlen=15)
        
        # Current state
        self.current_velocity = 0.0
        self.classification = "ANALYZING"
        self.confidence = 0.0
        self.total_frames = 0
        self.running_frames = 0
        self.not_running_frames = 0
        self.analyzing_frames = 0
        
        # For overall activity analysis
        self.activity_history = []  # Store (frame, classification, velocity, confidence)
        self.first_frame = None
        self.last_frame = None
    
    def update(self, center_x, center_y, keypoints, frame_number, fps):
        """Update tracker with new detection"""
        self.positions.append((center_x, center_y, frame_number))
        self.keypoints_history.append(keypoints)
        self.total_frames += 1
        
        # Track frame range
        if self.first_frame is None:
            self.first_frame = frame_number
        self.last_frame = frame_number
        
        # Calculate velocity
        self.current_velocity = self.calculate_velocity(fps)
        self.velocities.append(self.current_velocity)
        
        # Classify movement
        self.classify_movement()
        
        # Record activity history
        self.activity_history.append({
            'frame': frame_number,
            'classification': self.classification,
            'velocity': self.current_velocity,
            'confidence': self.confidence
        })
        
        # Update frame counters
        if self.classification == "RUNNING":
            self.running_frames += 1
        elif self.classification == "NOT RUNNING":
            self.not_running_frames += 1
        else:  # ANALYZING
            self.analyzing_frames += 1
    
    def calculate_velocity(self, fps):
        """Calculate movement velocity"""
        if len(self.positions) < 3:
            return 0.0
        
        # Get recent positions
        recent_positions = list(self.positions)[-8:]  # Last 8 positions
        
        if len(recent_positions) < 3:
            return 0.0
        
        # Calculate velocities between consecutive frames
        velocities = []
        for i in range(1, len(recent_positions)):
            x1, y1, f1 = recent_positions[i-1]
            x2, y2, f2 = recent_positions[i]
            
            # Time difference
            dt = (f2 - f1) / fps
            if dt > 0:
                # Distance
                dx = x2 - x1
                dy = y2 - y1
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Velocity
                velocity = distance / dt
                velocities.append(velocity)
        
        if not velocities:
            return 0.0
        
        # Return median velocity to reduce noise
        return np.median(velocities)
    
    def classify_movement(self):
        """Classify as RUNNING, NOT RUNNING, or ANALYZING"""
        if self.total_frames < MIN_FRAMES_FOR_ANALYSIS:
            self.classification = "ANALYZING"
            self.confidence = 0.0
            return
        
        # Get average velocity over recent frames
        recent_velocities = list(self.velocities)[-5:]  # Last 5 velocities
        avg_velocity = np.mean(recent_velocities) if recent_velocities else 0.0
        
        # Binary classification: RUNNING or NOT RUNNING
        if avg_velocity >= VELOCITY_THRESHOLD:
            self.classification = "RUNNING"
            # Confidence based on how much above threshold
            self.confidence = min(0.95, 0.6 + (avg_velocity - VELOCITY_THRESHOLD) / VELOCITY_THRESHOLD * 0.35)
        else:
            self.classification = "NOT RUNNING"
            # Confidence based on how much below threshold
            self.confidence = min(0.95, 0.7 + (VELOCITY_THRESHOLD - avg_velocity) / VELOCITY_THRESHOLD * 0.25)
        
        # Ensure minimum confidence
        self.confidence = max(0.3, self.confidence)
    
    def get_overall_activity_summary(self, fps):
        """Get overall activity classification for the entire tracking period"""
        if not self.activity_history:
            return {
                'overall_classification': 'UNKNOWN',
                'confidence': 0.0,
                'running_percentage': 0.0,
                'total_time': 0.0,
                'running_time': 0.0,
                'not_running_time': 0.0,
                'analyzing_time': 0.0
            }
        
        # Calculate time durations
        total_time = (self.last_frame - self.first_frame) / fps if self.first_frame and self.last_frame else 0.0
        running_time = self.running_frames / fps
        not_running_time = self.not_running_frames / fps
        analyzing_time = self.analyzing_frames / fps
        
        # Calculate percentages
        running_percentage = (self.running_frames / self.total_frames) * 100 if self.total_frames > 0 else 0
        not_running_percentage = (self.not_running_frames / self.total_frames) * 100 if self.total_frames > 0 else 0
        analyzing_percentage = (self.analyzing_frames / self.total_frames) * 100 if self.total_frames > 0 else 0
        
        # Determine overall classification based on dominant activity
        # Only classify as runner if they spent significant time running (>30% of trackable time)
        trackable_frames = self.running_frames + self.not_running_frames
        if trackable_frames > 0:
            running_ratio = self.running_frames / trackable_frames
            if running_ratio >= 0.3:  # 30% threshold
                overall_classification = "PREDOMINANTLY RUNNING"
                confidence = min(0.95, 0.5 + running_ratio * 0.45)
            elif running_ratio >= 0.1:  # 10% threshold
                overall_classification = "OCCASIONALLY RUNNING"
                confidence = min(0.85, 0.4 + running_ratio * 0.45)
            else:
                overall_classification = "MOSTLY STATIONARY/WALKING"
                confidence = min(0.9, 0.6 + (1 - running_ratio) * 0.3)
        else:
            overall_classification = "INSUFFICIENT DATA"
            confidence = 0.0
        
        return {
            'overall_classification': overall_classification,
            'confidence': confidence,
            'running_percentage': running_percentage,
            'not_running_percentage': not_running_percentage,
            'analyzing_percentage': analyzing_percentage,
            'total_time': total_time,
            'running_time': running_time,
            'not_running_time': not_running_time,
            'analyzing_time': analyzing_time,
            'max_velocity': max(self.velocities) if self.velocities else 0.0,
            'avg_velocity': np.mean(self.velocities) if self.velocities else 0.0,
            'frames_tracked': f"{self.first_frame}-{self.last_frame}" if self.first_frame and self.last_frame else "N/A"
        }

def get_keypoint_position(keypoints, keypoint_index):
    """Get position of specific keypoint"""
    if keypoint_index < len(keypoints):
        kpt = keypoints[keypoint_index]
        if len(kpt) >= 2:
            x, y = kpt[0], kpt[1]
            if not (np.isnan(x) or np.isnan(y)) and x > 0 and y > 0:
                return (float(x), float(y))
    return None

def calculate_center_of_mass(keypoints):
    """Calculate center of mass from keypoints"""
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

def draw_person_info(frame, person, bbox):
    """Draw person information on frame"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Choose color based on classification - ONLY 3 STATES
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
    
    # Prepare text
    status_text = f"ID:{person.track_id} - {person.classification}"
    velocity_text = f"Velocity: {person.current_velocity:.1f} px/s"
    confidence_text = f"Confidence: {person.confidence:.2f}"
    
    # Draw text background
    text_y = y1 - 15
    cv2.rectangle(frame, (x1, text_y - 50), (x1 + 350, text_y + 10), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, status_text, (x1 + 5, text_y - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, velocity_text, (x1 + 5, text_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, confidence_text, (x1 + 5, text_y + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw movement trail
    if SHOW_TRAILS and len(person.positions) > 1:
        positions = list(person.positions)[-TRAIL_LENGTH:]
        for i in range(1, len(positions)):
            x1_trail, y1_trail, _ = positions[i-1]
            x2_trail, y2_trail, _ = positions[i]
            
            # Fade trail color
            alpha = i / len(positions)
            trail_color = tuple(int(c * alpha) for c in color)
            
            cv2.line(frame, (int(x1_trail), int(y1_trail)), 
                    (int(x2_trail), int(y2_trail)), trail_color, 2)

def draw_statistics(frame, persons, frame_count, total_frames, fps):
    """Draw system statistics"""
    height, width = frame.shape[:2]
    
    # Background for statistics
    cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)
    
    # Count people by status - ONLY 3 CATEGORIES
    running_count = sum(1 for p in persons.values() if p.classification == "RUNNING")
    not_running_count = sum(1 for p in persons.values() if p.classification == "NOT RUNNING")
    analyzing_count = sum(1 for p in persons.values() if p.classification == "ANALYZING")
    
    # Progress
    progress_percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
    time_in_video = frame_count / fps
    
    # Statistics text - SIMPLIFIED TO 3 CATEGORIES
    stats_text = f"🔴 RUNNING: {running_count} | 🟢 NOT RUNNING: {not_running_count} | 🟡 ANALYZING: {analyzing_count}"
    cv2.putText(frame, stats_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    progress_text = f"Progress: {progress_percent:.1f}% | Time: {time_in_video:.1f}s | Frame: {frame_count}/{total_frames}"
    cv2.putText(frame, progress_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    threshold_text = f"Velocity Threshold: {VELOCITY_THRESHOLD:.0f} px/s | Press 'q' to quit"
    cv2.putText(frame, threshold_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

def main():
    """Main function"""
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
    
    # Initialize video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    # Person trackers
    persons = {}
    frame_count = 0
    
    # Statistics
    total_running_detections = 0
    total_alerts = 0

    print("📋 Classification: RUNNING | NOT RUNNING | ANALYZING")
    print("Press 'q' to quit early")

    start_time = time.time()
    
    try:
        # Process video frame by frame
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
                    
                    # Try to use center of mass from keypoints, fallback to bbox center
                    center_pos = calculate_center_of_mass(keypoints) if keypoints is not None else None
                    if center_pos:
                        center_x, center_y = center_pos
                    else:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                    
                    # Initialize or update person tracker
                    if track_id not in persons:
                        persons[track_id] = SimplePersonTracker(track_id)
                    
                    # Update tracker
                    persons[track_id].update(center_x, center_y, keypoints, frame_count, fps)
                    
                    # Count running detections
                    if persons[track_id].classification == "RUNNING":
                        total_running_detections += 1
                        
                        # Simple alert (print to console)
                        if frame_count % 30 == 0:  # Every 30 frames (1 second)
                            time_in_video = frame_count / fps
                            print(f"🚨 RUNNING DETECTED! Person {track_id} at {time_in_video:.1f}s (Velocity: {persons[track_id].current_velocity:.1f} px/s)")
                            total_alerts += 1
                    
                    # Draw person information
                    draw_person_info(frame, persons[track_id], bbox)
                    
                    # Draw keypoints
                    if keypoints is not None:
                        draw_keypoints(frame, keypoints)
            
            # Clean up old persons (not seen for 60 frames)
            current_ids = set()
            if result.boxes is not None and result.boxes.id is not None:
                current_ids = set(result.boxes.id.cpu().numpy().astype(int))
            
            persons_to_remove = []
            for track_id, person in persons.items():
                if track_id not in current_ids:
                    if len(person.positions) > 0:
                        last_seen_frame = person.positions[-1][2]
                        if frame_count - last_seen_frame > 60:  # 2 seconds
                            persons_to_remove.append(track_id)
            
            for track_id in persons_to_remove:
                del persons[track_id]
            
            # Draw statistics
            draw_statistics(frame, persons, frame_count, total_frames, fps)
            
            # Write frame to output video
            out.write(frame)
            
            # Display frame
            cv2.imshow('Running Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Progress update every 60 frames
            if frame_count % 60 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed_time = time.time() - start_time
                processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"📊 Progress: {progress:.1f}% | Processing: {processing_fps:.1f} FPS | Running detections: {total_running_detections}")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Final report with improved summary
        processing_time = time.time() - start_time
        video_duration = total_frames / fps
        
        print(f"\n" + "="*80)
        print(f"🎬 VIDEO ANALYSIS COMPLETE")
        print(f"="*80)
        print(f"📁 Input: {VIDEO_PATH}")
        print(f"📁 Output: {OUTPUT_PATH}")
        print(f"⏱️  Video Duration: {video_duration:.1f}s | Processing Time: {processing_time:.1f}s")
        print(f"📊 Total Frames Processed: {frame_count}/{total_frames}")
        print(f"👥 Total Persons Detected: {len(persons)}")
        print(f"🏃 Total Running Detections: {total_running_detections}")
        print(f"🚨 Total Alerts Generated: {total_alerts}")
        
     

if __name__ == "__main__":
    main()