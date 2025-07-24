from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import imageio
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
from scipy import signal
from scipy.interpolate import interp1d

# ---------------- ADAPTIVE CONFIG ----------------
video_path = "running_person.mp4"
output_path = "output_running.mp4"
model_path = "yolov8n-pose.pt"

# Analysis parameters
ANALYSIS_WINDOW = 60
MIN_DETECTION_FRAMES = 25
CONFIDENCE_THRESHOLD = 0.6

# ADAPTIVE thresholds - will be calculated based on video characteristics
# These are base values that will be scaled
BASE_WALKING_VELOCITY = 8.0    # Base walking speed in px/s
BASE_RUNNING_VELOCITY = 15.0   # Base running speed in px/s
VELOCITY_SCALE_FACTOR = 1.0    # Will be calculated automatically

# Frequency thresholds (more universal)
MIN_STRIDE_FREQUENCY = 1.6     # Lower for slow jogging
RUNNING_FREQUENCY_RANGE = (1.8, 4.5)  # Wider range

# Other thresholds
MIN_VERTICAL_OSCILLATION = 0.08  # Lower threshold
MAX_GROUND_CONTACT = 0.60       # More lenient
MIN_RHYTHM_CONSISTENCY = 0.25   # Lower threshold

DEBUG_MODE = True

@dataclass
class VideoCharacteristics:
    """Automatically detected video characteristics for adaptive thresholds"""
    fps: float
    width: int
    height: int
    avg_person_height: float = 0.0
    avg_person_width: float = 0.0
    movement_scale: float = 1.0
    velocity_scale: float = 1.0
    
    def __post_init__(self):
        # Estimate scale based on resolution
        # Higher resolution = more pixels per meter = higher velocity values
        resolution_factor = math.sqrt(self.width * self.height) / 1000.0  # Normalize to ~1000px base
        self.velocity_scale = max(0.3, min(3.0, resolution_factor))  # Clamp between 0.3x and 3x

@dataclass
class GaitMetrics:
    stride_length: float = 0.0
    stride_frequency: float = 0.0
    vertical_oscillation: float = 0.0
    ground_contact_time: float = 0.0
    flight_time: float = 0.0
    cadence: float = 0.0
    knee_drive: float = 0.0
    arm_swing_amplitude: float = 0.0
    movement_velocity: float = 0.0
    movement_velocity_smooth: float = 0.0
    movement_velocity_peak: float = 0.0
    rhythm_consistency: float = 0.0
    body_lean: float = 0.0
    step_regularity: float = 0.0
    partial_visibility_ratio: float = 1.0
    
    # Adaptive thresholds used
    adaptive_walking_threshold: float = 0.0
    adaptive_running_threshold: float = 0.0
    
    # Debug information
    velocity_debug: dict = None
    stage_results: dict = None
    
    def __post_init__(self):
        if self.velocity_debug is None:
            self.velocity_debug = {}
        if self.stage_results is None:
            self.stage_results = {}

@dataclass
class PersonTracker:
    track_id: int
    keypoint_history: deque
    position_history: deque
    gait_metrics_history: deque
    frame_timestamps: deque
    velocity_history: deque
    visibility_history: deque
    final_classification: str = "ANALYZING"
    confidence: float = 0.0
    analysis_complete: bool = False
    last_foot_contact: Dict = None
    stride_count: int = 0
    current_gait_cycle: List = None
    
    # Person-specific characteristics
    estimated_height: float = 0.0
    estimated_width: float = 0.0
    personal_velocity_scale: float = 1.0

    def __post_init__(self):
        if self.last_foot_contact is None:
            self.last_foot_contact = {"left": None, "right": None}
        if self.current_gait_cycle is None:
            self.current_gait_cycle = []

class AdaptiveRunningDetection:
    """Adaptive running detection that scales to video characteristics"""
    
    def __init__(self, video_chars: VideoCharacteristics):
        self.video_chars = video_chars
        self.fps = video_chars.fps
        
        # Calculate adaptive thresholds
        self.walking_velocity_threshold = BASE_WALKING_VELOCITY * video_chars.velocity_scale
        self.running_velocity_threshold = BASE_RUNNING_VELOCITY * video_chars.velocity_scale
        
        print(f"🎯 Adaptive Thresholds:")
        print(f"   Walking velocity: {self.walking_velocity_threshold:.1f} px/s")
        print(f"   Running velocity: {self.running_velocity_threshold:.1f} px/s")
        print(f"   Scale factor: {video_chars.velocity_scale:.2f}")
        
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
            if idx < len(keypoints) and len(keypoints[idx]) >= 2:
                x, y = keypoints[idx][:2]
                if not (np.isnan(x) or np.isnan(y)) and x > 0 and y > 0:
                    return (float(x), float(y))
        except (IndexError, KeyError):
            pass
        return None

    def calculate_visibility_ratio(self, keypoints: np.ndarray) -> float:
        """Calculate visibility ratio"""
        visible_count = 0
        total_count = len(self.keypoint_names)
        
        for name in self.keypoint_names:
            if self.get_valid_keypoint(keypoints, name):
                visible_count += 1
        
        return visible_count / total_count

    def calculate_center_of_mass(self, keypoints: np.ndarray) -> Optional[Tuple[float, float]]:
        """Calculate center of mass with multiple fallbacks"""
        priority_groups = [
            ['left_hip', 'right_hip'],
            ['left_shoulder', 'right_shoulder'],
            ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder'],
            ['nose'],
        ]
        
        for group in priority_groups:
            valid_points = []
            for point_name in group:
                point = self.get_valid_keypoint(keypoints, point_name)
                if point:
                    valid_points.append(point)
            
            if valid_points:
                if len(valid_points) == 1:
                    return valid_points[0]
                else:
                    avg_x = sum(p[0] for p in valid_points) / len(valid_points)
                    avg_y = sum(p[1] for p in valid_points) / len(valid_points)
                    return (avg_x, avg_y)
        
        return None

    def calculate_movement_velocity_enhanced(self, person: PersonTracker) -> Dict:
        """Enhanced velocity calculation with multiple methods and debugging"""
        if len(person.position_history) < 8:
            return {
                'velocity': 0.0,
                'velocity_smooth': 0.0,
                'velocity_peak': 0.0,
                'debug': {'error': 'insufficient_positions', 'count': len(person.position_history)}
            }
        
        positions = list(person.position_history)
        debug_info = {'position_count': len(positions)}
        
        # Method 1: Frame-to-frame velocity
        frame_velocities = []
        for i in range(1, len(positions)):
            dt = 1.0 / self.fps
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            velocity = math.sqrt(dx*dx + dy*dy) / dt
            frame_velocities.append(velocity)
        
        # Method 2: Multi-window velocities for smoothing
        window_velocities = []
        for window_size in [3, 5, 7]:
            if len(positions) > window_size:
                for i in range(window_size, len(positions)):
                    dt = window_size / self.fps
                    dx = positions[i][0] - positions[i-window_size][0]
                    dy = positions[i][1] - positions[i-window_size][1]
                    velocity = math.sqrt(dx*dx + dy*dy) / dt
                    window_velocities.append(velocity)
        
        # Method 3: Directional velocity (horizontal movement focus)
        horizontal_velocities = []
        for i in range(1, len(positions)):
            dt = 1.0 / self.fps
            dx = abs(positions[i][0] - positions[i-1][0])  # Horizontal only
            velocity = dx / dt
            horizontal_velocities.append(velocity)
        
        # Calculate different velocity metrics
        if frame_velocities:
            velocity_mean = np.mean(frame_velocities)
            velocity_median = np.median(frame_velocities)
            velocity_75th = np.percentile(frame_velocities, 75)
            velocity_90th = np.percentile(frame_velocities, 90)
            velocity_max = np.max(frame_velocities)
            
            debug_info.update({
                'frame_velocities_count': len(frame_velocities),
                'velocity_mean': velocity_mean,
                'velocity_median': velocity_median,
                'velocity_75th': velocity_75th,
                'velocity_90th': velocity_90th,
                'velocity_max': velocity_max
            })
        else:
            velocity_mean = velocity_median = velocity_75th = velocity_90th = velocity_max = 0.0
        
        if window_velocities:
            velocity_smooth = np.median(window_velocities)
            debug_info['window_velocities_count'] = len(window_velocities)
            debug_info['velocity_smooth'] = velocity_smooth
        else:
            velocity_smooth = velocity_median
        
        if horizontal_velocities:
            velocity_horizontal = np.percentile(horizontal_velocities, 75)
            debug_info['velocity_horizontal'] = velocity_horizontal
        else:
            velocity_horizontal = 0.0
        
        # Choose best velocity estimate
        # For running detection, we want to capture peak movement but avoid outliers
        final_velocity = max(velocity_75th, velocity_smooth)  # Use higher of 75th percentile or smooth
        velocity_peak = velocity_90th  # Peak velocity for burst detection
        
        # Adaptive scaling based on person size if available
        if person.estimated_height > 0:
            # Larger people in frame = closer to camera = higher pixel velocities
            size_factor = person.estimated_height / 120.0  # Normalize to ~120px person
            size_factor = max(0.5, min(2.0, size_factor))  # Clamp
            debug_info['size_factor'] = size_factor
        else:
            size_factor = 1.0
        
        # Apply person-specific scaling
        final_velocity *= person.personal_velocity_scale * size_factor
        velocity_smooth *= person.personal_velocity_scale * size_factor
        velocity_peak *= person.personal_velocity_scale * size_factor
        
        debug_info.update({
            'final_velocity': final_velocity,
            'personal_scale': person.personal_velocity_scale,
            'size_factor': size_factor
        })
        
        return {
            'velocity': final_velocity,
            'velocity_smooth': velocity_smooth,
            'velocity_peak': velocity_peak,
            'debug': debug_info
        }

    def update_person_characteristics(self, person: PersonTracker):
        """Update person-specific characteristics for better scaling"""
        if len(person.keypoint_history) < 10:
            return
        
        # Estimate person size in the frame
        heights = []
        widths = []
        
        for keypoints in list(person.keypoint_history)[-10:]:  # Use recent frames
            # Height estimation
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
                if point and (foot_y is None or point[1] > foot_y):
                    foot_y = point[1]
            
            if head_y and foot_y and foot_y > head_y:
                height = foot_y - head_y
                if 50 < height < 400:
                    heights.append(height)
            
            # Width estimation (shoulder span)
            left_shoulder = self.get_valid_keypoint(keypoints, 'left_shoulder')
            right_shoulder = self.get_valid_keypoint(keypoints, 'right_shoulder')
            if left_shoulder and right_shoulder:
                width = abs(right_shoulder[0] - left_shoulder[0])
                if 20 < width < 200:
                    widths.append(width)
        
        if heights:
            person.estimated_height = np.median(heights)
            # Update personal velocity scale based on size
            # Larger people in frame = closer = higher pixel velocities expected
            base_height = 120.0  # Expected height for medium distance
            person.personal_velocity_scale = max(0.5, min(2.0, person.estimated_height / base_height))
        
        if widths:
            person.estimated_width = np.median(widths)

    def detect_stride_pattern_improved(self, person: PersonTracker) -> Dict:
        """Improved stride detection with better signal processing"""
        if len(person.keypoint_history) < 20:
            return {"frequency": 0.0, "consistency": 0.0, "detected_strides": 0, "quality": "insufficient_data"}

        keypoint_data = list(person.keypoint_history)
        
        # Multiple signals for robustness
        signals_data = {
            'vertical_movement': [],
            'ankle_alternation': [],
            'knee_separation': [],
            'hip_oscillation': []
        }

        for keypoints in keypoint_data:
            # 1. Vertical movement of center of mass
            com = self.calculate_center_of_mass(keypoints)
            if com:
                signals_data['vertical_movement'].append(com[1])
            else:
                signals_data['vertical_movement'].append(np.nan)

            # 2. Ankle alternation pattern
            left_ankle = self.get_valid_keypoint(keypoints, 'left_ankle')
            right_ankle = self.get_valid_keypoint(keypoints, 'right_ankle')
            if left_ankle and right_ankle:
                alternation = left_ankle[1] - right_ankle[1]
                signals_data['ankle_alternation'].append(alternation)
            else:
                signals_data['ankle_alternation'].append(np.nan)

            # 3. Knee separation
            left_knee = self.get_valid_keypoint(keypoints, 'left_knee')
            right_knee = self.get_valid_keypoint(keypoints, 'right_knee')
            if left_knee and right_knee:
                separation = abs(left_knee[0] - right_knee[0])
                signals_data['knee_separation'].append(separation)
            else:
                signals_data['knee_separation'].append(np.nan)

            # 4. Hip oscillation
            left_hip = self.get_valid_keypoint(keypoints, 'left_hip')
            right_hip = self.get_valid_keypoint(keypoints, 'right_hip')
            if left_hip and right_hip:
                hip_center_y = (left_hip[1] + right_hip[1]) / 2
                signals_data['hip_oscillation'].append(hip_center_y)
            else:
                signals_data['hip_oscillation'].append(np.nan)

        detected_frequencies = []
        consistency_scores = []
        signal_qualities = {}

        for signal_name, signal_values in signals_data.items():
            valid_indices = ~np.isnan(signal_values)
            valid_ratio = np.sum(valid_indices) / len(signal_values)
            
            if valid_ratio < 0.6:
                signal_qualities[signal_name] = f"insufficient_data_{valid_ratio:.2f}"
                continue

            valid_values = np.array(signal_values)[valid_indices]
            if len(valid_values) < 15:
                signal_qualities[signal_name] = f"too_few_points_{len(valid_values)}"
                continue

            # Interpolate missing values if needed
            if valid_ratio < 0.9:
                valid_times = np.arange(len(signal_values))[valid_indices]
                interp_times = np.arange(len(signal_values))
                f = interp1d(valid_times, valid_values, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
                signal_interp = f(interp_times)
            else:
                signal_interp = np.array(signal_values)
                signal_interp[np.isnan(signal_interp)] = np.nanmean(signal_interp)

            # Improved signal processing
            signal_detrended = signal.detrend(signal_interp)
            
            # Apply windowing
            windowed = signal_detrended * signal.windows.hann(len(signal_detrended))

            # FFT analysis
            fft = np.fft.fft(windowed)
            freqs = np.fft.fftfreq(len(windowed), 1.0/self.fps)

            # Human gait frequency range
            pos_mask = (freqs > 0.8) & (freqs < 5.0)
            if not np.any(pos_mask):
                signal_qualities[signal_name] = "no_valid_frequencies"
                continue

            pos_freqs = freqs[pos_mask]
            pos_power = np.abs(fft[pos_mask])

            peak_idx = np.argmax(pos_power)
            dominant_freq = pos_freqs[peak_idx]

            # More lenient consistency calculation
            peak_power = pos_power[peak_idx]
            mean_power = np.mean(pos_power)
            consistency = (peak_power - mean_power) / (peak_power + mean_power + 1e-6)

            # Accept reasonable patterns
            if consistency > 0.2:  # More lenient threshold
                detected_frequencies.append(dominant_freq)
                consistency_scores.append(consistency)
                signal_qualities[signal_name] = f"freq_{dominant_freq:.2f}_cons_{consistency:.3f}"
            else:
                signal_qualities[signal_name] = f"low_consistency_{consistency:.3f}"

        # Combine results
        if detected_frequencies:
            final_frequency = np.median(detected_frequencies)
            final_consistency = np.mean(consistency_scores)
            stride_count = len(detected_frequencies)
            quality = "good"
        else:
            final_frequency = 0.0
            final_consistency = 0.0
            stride_count = 0
            quality = "no_clear_pattern"

        return {
            "frequency": final_frequency,
            "consistency": final_consistency,
            "detected_strides": stride_count,
            "individual_frequencies": detected_frequencies,
            "quality": quality,
            "signal_qualities": signal_qualities
        }

    def calculate_enhanced_metrics(self, person: PersonTracker) -> GaitMetrics:
        """Calculate enhanced metrics with adaptive thresholds"""
        if len(person.keypoint_history) < 15:
            return GaitMetrics()

        # Update person characteristics
        self.update_person_characteristics(person)

        metrics = GaitMetrics()
        keypoint_data = list(person.keypoint_history)
        position_data = list(person.position_history)

        # Set adaptive thresholds
        metrics.adaptive_walking_threshold = self.walking_velocity_threshold
        metrics.adaptive_running_threshold = self.running_velocity_threshold

        # Visibility ratio
        visibility_ratios = []
        for keypoints in keypoint_data:
            visibility_ratios.append(self.calculate_visibility_ratio(keypoints))
        metrics.partial_visibility_ratio = np.mean(visibility_ratios)

        # Enhanced velocity calculation
        velocity_result = self.calculate_movement_velocity_enhanced(person)
        metrics.movement_velocity = velocity_result['velocity']
        metrics.movement_velocity_smooth = velocity_result['velocity_smooth']
        metrics.movement_velocity_peak = velocity_result['velocity_peak']
        metrics.velocity_debug = velocity_result['debug']

        # Stride analysis
        stride_analysis = self.detect_stride_pattern_improved(person)
        metrics.stride_frequency = stride_analysis["frequency"]
        metrics.rhythm_consistency = stride_analysis["consistency"]
        metrics.stage_results["stride_analysis"] = stride_analysis

        # Vertical oscillation
        if len(position_data) >= 15:
            y_positions = [pos[1] for pos in position_data if pos]
            if len(y_positions) >= 10:
                y_range = np.percentile(y_positions, 85) - np.percentile(y_positions, 15)
                if person.estimated_height > 0:
                    metrics.vertical_oscillation = y_range / person.estimated_height
                else:
                    # Fallback estimation
                    estimated_height = max(100, np.percentile(y_positions, 90) - np.percentile(y_positions, 10))
                    metrics.vertical_oscillation = y_range / estimated_height

        # Ground contact analysis
        ground_analysis = self.analyze_ground_contact_adaptive(keypoint_data)
        metrics.ground_contact_time = ground_analysis.get('avg_contact_ratio', 0.5)

        # Cadence
        if metrics.stride_frequency > 0:
            metrics.cadence = metrics.stride_frequency * 60 * 2

        return metrics

    def analyze_ground_contact_adaptive(self, keypoint_history: List) -> Dict:
        """Adaptive ground contact analysis"""
        all_ankle_positions = []
        
        for keypoints in keypoint_history:
            for side in ['left', 'right']:
                ankle = self.get_valid_keypoint(keypoints, f'{side}_ankle')
                if ankle:
                    all_ankle_positions.append(ankle[1])
        
        if len(all_ankle_positions) < 8:
            return {'avg_contact_ratio': 0.5, 'quality': 'insufficient_data'}

        # Adaptive ground level
        ground_level = np.percentile(all_ankle_positions, 85)
        ground_threshold = np.std(all_ankle_positions) * 0.4

        contact_ratios = []
        for keypoints in keypoint_history:
            frame_contacts = []
            for side in ['left', 'right']:
                ankle = self.get_valid_keypoint(keypoints, f'{side}_ankle')
                if ankle:
                    is_contact = abs(ankle[1] - ground_level) < ground_threshold
                    frame_contacts.append(1.0 if is_contact else 0.0)
            
            if frame_contacts:
                contact_ratios.append(np.mean(frame_contacts))

        avg_contact = np.mean(contact_ratios) if contact_ratios else 0.5
        return {'avg_contact_ratio': avg_contact, 'quality': 'good'}

    def classify_running_adaptive(self, metrics: GaitMetrics) -> Tuple[str, float]:
        """Adaptive classification with scaled thresholds"""
        
        stage_results = {}
        debug_scores = {}
        
        # Stage 1: Basic movement (MUST PASS)
        stage1_pass = False
        stage1_reasons = []
        
        # Use adaptive thresholds
        min_movement = metrics.adaptive_walking_threshold
        
        if metrics.movement_velocity >= min_movement:
            stage1_reasons.append(f"✓ Velocity: {metrics.movement_velocity:.1f} >= {min_movement:.1f}")
        else:
            stage1_reasons.append(f"✗ Velocity: {metrics.movement_velocity:.1f} < {min_movement:.1f}")
        
        if metrics.stride_frequency >= MIN_STRIDE_FREQUENCY:
            stage1_reasons.append(f"✓ Frequency: {metrics.stride_frequency:.2f} >= {MIN_STRIDE_FREQUENCY}")
        else:
            stage1_reasons.append(f"✗ Frequency: {metrics.stride_frequency:.2f} < {MIN_STRIDE_FREQUENCY}")
        
        if (metrics.movement_velocity >= min_movement and 
            metrics.stride_frequency >= MIN_STRIDE_FREQUENCY):
            stage1_pass = True
        
        stage_results['stage1'] = {'pass': stage1_pass, 'reasons': stage1_reasons}
        
        if not stage1_pass:
            metrics.stage_results['classification'] = stage_results
            return "NOT RUNNING", 0.85
        
        # Stage 2: Running-specific (FLEXIBLE SCORING)
        stage2_score = 0.0
        stage2_max = 4.0
        stage2_reasons = []
        
        # Adaptive running velocity check
        running_threshold = metrics.adaptive_running_threshold
        if metrics.movement_velocity >= running_threshold:
            stage2_score += 1.0
            stage2_reasons.append(f"✓ Running velocity: {metrics.movement_velocity:.1f} >= {running_threshold:.1f}")
        elif metrics.movement_velocity >= running_threshold * 0.8:
            # Partial credit for close velocities
            partial_score = 0.6
            stage2_score += partial_score
            stage2_reasons.append(f"~ Running velocity: {metrics.movement_velocity:.1f} >= {running_threshold*0.8:.1f} (partial)")
        else:
            stage2_reasons.append(f"✗ Running velocity: {metrics.movement_velocity:.1f} < {running_threshold:.1f}")
        
        # Peak velocity check (alternative indicator)
        if metrics.movement_velocity_peak >= running_threshold * 1.2:
            stage2_score += 0.5  # Bonus for peak velocity
            stage2_reasons.append(f"✓ Peak velocity bonus: {metrics.movement_velocity_peak:.1f}")
        
        # Frequency range check
        if RUNNING_FREQUENCY_RANGE[0] <= metrics.stride_frequency <= RUNNING_FREQUENCY_RANGE[1]:
            stage2_score += 1.0
            stage2_reasons.append(f"✓ Running frequency: {metrics.stride_frequency:.2f} in {RUNNING_FREQUENCY_RANGE}")
        elif metrics.stride_frequency >= RUNNING_FREQUENCY_RANGE[0] * 0.9:
            stage2_score += 0.5
            stage2_reasons.append(f"~ Running frequency: {metrics.stride_frequency:.2f} close to range")
        else:
            stage2_reasons.append(f"✗ Running frequency: {metrics.stride_frequency:.2f} not in {RUNNING_FREQUENCY_RANGE}")
        
        # Vertical oscillation check
        if metrics.vertical_oscillation >= MIN_VERTICAL_OSCILLATION:
            stage2_score += 1.0
            stage2_reasons.append(f"✓ Vertical oscillation: {metrics.vertical_oscillation:.3f} >= {MIN_VERTICAL_OSCILLATION}")
        elif metrics.vertical_oscillation >= MIN_VERTICAL_OSCILLATION * 0.7:
            stage2_score += 0.5
            stage2_reasons.append(f"~ Vertical oscillation: {metrics.vertical_oscillation:.3f} partial")
        else:
            stage2_reasons.append(f"✗ Vertical oscillation: {metrics.vertical_oscillation:.3f} < {MIN_VERTICAL_OSCILLATION}")
        
        # Ground contact check
        if metrics.ground_contact_time <= MAX_GROUND_CONTACT:
            stage2_score += 1.0
            stage2_reasons.append(f"✓ Ground contact: {metrics.ground_contact_time:.3f} <= {MAX_GROUND_CONTACT}")
        else:
            stage2_reasons.append(f"✗ Ground contact: {metrics.ground_contact_time:.3f} > {MAX_GROUND_CONTACT}")
        
        stage2_pass = stage2_score >= 2.5  # More lenient: 2.5 out of 4
        stage_results['stage2'] = {'pass': stage2_pass, 'score': stage2_score, 'max': stage2_max, 'reasons': stage2_reasons}
        
        # Stage 3: Consistency (BONUS STAGE)
        stage3_score = 0.0
        stage3_max = 2.0
        stage3_reasons = []
        
        if metrics.rhythm_consistency >= MIN_RHYTHM_CONSISTENCY:
            stage3_score += 1.0
            stage3_reasons.append(f"✓ Rhythm consistency: {metrics.rhythm_consistency:.3f} >= {MIN_RHYTHM_CONSISTENCY}")
        else:
            stage3_reasons.append(f"✗ Rhythm consistency: {metrics.rhythm_consistency:.3f} < {MIN_RHYTHM_CONSISTENCY}")
        
        if metrics.partial_visibility_ratio >= 0.6:
            stage3_score += 1.0
            stage3_reasons.append(f"✓ Visibility: {metrics.partial_visibility_ratio:.1%} >= 60%")
        else:
            stage3_reasons.append(f"✗ Visibility: {metrics.partial_visibility_ratio:.1%} < 60%")
        
        stage3_pass = stage3_score >= 1.0
        stage_results['stage3'] = {'pass': stage3_pass, 'score': stage3_score, 'max': stage3_max, 'reasons': stage3_reasons}
        
        # FINAL CLASSIFICATION with adaptive scoring
        base_score = (stage2_score / stage2_max) * 0.8
        bonus_score = (stage3_score / stage3_max) * 0.2
        total_score = base_score + bonus_score
        
        # Visibility boost
        visibility_boost = min(metrics.partial_visibility_ratio * 1.1, 1.0)
        final_score = total_score * visibility_boost
        
        stage_results['final'] = {
            'base_score': base_score,
            'bonus_score': bonus_score,
            'total_score': total_score,
            'visibility_boost': visibility_boost,
            'final_score': final_score
        }
        
        metrics.stage_results['classification'] = stage_results
        
        # More lenient classification thresholds
        if final_score >= 0.65 and stage2_pass:
            return "RUNNING", min(0.95, final_score)
        elif final_score >= 0.45 and stage2_score >= 2.0:
            return "RUNNING", min(0.8, final_score)
        else:
            return "NOT RUNNING", max(0.1, 1.0 - final_score)

# Initialize video characteristics
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found: {video_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# Create video characteristics for adaptive scaling
video_chars = VideoCharacteristics(fps=fps, width=width, height=height)

print(f"🎥 Input: {width}x{height} @ {fps:.1f} FPS ({total_frames} frames)")
print(f"🎯 Video Scale Factor: {video_chars.velocity_scale:.2f}")

# Initialize adaptive algorithm
model = YOLO(model_path)
algorithm = AdaptiveRunningDetection(video_chars)
writer = imageio.get_writer(output_path, fps=fps)
persons: Dict[int, PersonTracker] = {}
frame_idx = 0

# Process video with adaptive thresholds
for r in model.track(source=video_path, stream=True, persist=True, verbose=False):
    frame = r.orig_img
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
    ids = r.boxes.id.cpu().numpy().astype(int) if (r.boxes and r.boxes.id is not None) else []
    kpts_all = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else []

    for i, track_id in enumerate(ids):
        x1, y1, x2, y2 = boxes[i]
        keypoints = kpts_all[i] if len(kpts_all) > i else np.full((17, 2), np.nan)

        # Initialize person tracking
        if track_id not in persons:
            persons[track_id] = PersonTracker(
                track_id=track_id,
                keypoint_history=deque(maxlen=ANALYSIS_WINDOW),
                position_history=deque(maxlen=ANALYSIS_WINDOW),
                gait_metrics_history=deque(maxlen=10),
                frame_timestamps=deque(maxlen=ANALYSIS_WINDOW),
                velocity_history=deque(maxlen=30),
                visibility_history=deque(maxlen=30),
            )

        person = persons[track_id]

        # Store data
        person.keypoint_history.append(keypoints)
        person.frame_timestamps.append(frame_idx)

        # Calculate visibility
        visibility = algorithm.calculate_visibility_ratio(keypoints)
        person.visibility_history.append(visibility)

        # Calculate center of mass
        center_of_mass = algorithm.calculate_center_of_mass(keypoints)
        if center_of_mass:
            person.position_history.append(center_of_mass)

        # Perform adaptive analysis
        if (len(person.keypoint_history) >= MIN_DETECTION_FRAMES and 
            not person.analysis_complete):
            
            # Calculate enhanced metrics
            metrics = algorithm.calculate_enhanced_metrics(person)
            person.gait_metrics_history.append(metrics)

            # Adaptive classification
            if len(person.gait_metrics_history) >= 2:
                latest_metrics = person.gait_metrics_history[-1]
                classification, confidence = algorithm.classify_running_adaptive(latest_metrics)
                
                person.final_classification = classification
                person.confidence = confidence
                person.analysis_complete = True

        # Color scheme: RED for running, GREEN for not running
        if person.analysis_complete:
            if person.final_classification == "RUNNING":
                color = (0, 0, 255)  # RED
                thickness = 4
            else:
                color = (0, 255, 0)  # GREEN
                thickness = 3
            
            status_text = f"ID:{track_id} - {person.final_classification} ({person.confidence:.2f})"
        else:
            color = (0, 255, 255)  # YELLOW for analyzing
            thickness = 2
            progress = len(person.keypoint_history) / MIN_DETECTION_FRAMES
            status_text = f"ID:{track_id} - ANALYZING {len(person.keypoint_history)}/{MIN_DETECTION_FRAMES}"

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # Draw status
        y_offset = int(y1) - 10
        cv2.putText(frame, status_text, (int(x1), y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show detailed velocity information
        if person.analysis_complete and person.gait_metrics_history and DEBUG_MODE:
            latest = person.gait_metrics_history[-1]
            
            # Velocity information
            y_offset -= 20
            vel_text = f"Vel:{latest.movement_velocity:.1f} (th:{latest.adaptive_running_threshold:.1f})"
            cv2.putText(frame, vel_text, (int(x1), y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Frequency and other metrics
            y_offset -= 15
            freq_text = f"Freq:{latest.stride_frequency:.1f}Hz Osc:{latest.vertical_oscillation:.2f}"
            cv2.putText(frame, freq_text, (int(x1), y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Person characteristics
            if person.estimated_height > 0:
                y_offset -= 15
                char_text = f"H:{person.estimated_height:.0f}px Scale:{person.personal_velocity_scale:.1f}"
                cv2.putText(frame, char_text, (int(x1), y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw keypoints
        for idx, kpt in enumerate(keypoints):
            if len(kpt) >= 2 and not (np.isnan(kpt[0]) or np.isnan(kpt[1])):
                x_kpt, y_kpt = int(kpt[0]), int(kpt[1])
                
                if idx in [11, 12]:  # Hips
                    cv2.circle(frame, (x_kpt, y_kpt), 6, (0, 255, 255), -1)
                elif idx in [13, 14]:  # Knees
                    cv2.circle(frame, (x_kpt, y_kpt), 5, (255, 255, 0), -1)
                elif idx in [15, 16]:  # Ankles
                    cv2.circle(frame, (x_kpt, y_kpt), 4, (255, 0, 255), -1)
                else:
                    cv2.circle(frame, (x_kpt, y_kpt), 2, (128, 128, 128), -1)

    # Statistics panel
    cv2.rectangle(frame, (0, 0), (width, 140), (0, 0, 0), -1)

    # Count classifications
    running_count = sum(1 for p in persons.values() 
                       if p.analysis_complete and p.final_classification == "RUNNING")
    not_running_count = sum(1 for p in persons.values() 
                           if p.analysis_complete and p.final_classification == "NOT RUNNING")
    analyzing_count = sum(1 for p in persons.values() if not p.analysis_complete)

    # Display statistics
    stats_text = f"🔴 RUNNING: {running_count} | 🟢 NOT RUNNING: {not_running_count} | 🟡 ANALYZING: {analyzing_count}"
    cv2.putText(frame, stats_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    frame_text = f"Frame: {frame_idx}/{total_frames} | ADAPTIVE Detection"
    cv2.putText(frame, frame_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Show adaptive thresholds
    thresh_text = f"Adaptive Thresholds: Walk≥{algorithm.walking_velocity_threshold:.1f} Run≥{algorithm.running_velocity_threshold:.1f} px/s"
    cv2.putText(frame, thresh_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    scale_text = f"Scale Factor: {video_chars.velocity_scale:.2f} | Resolution: {width}x{height}"
    cv2.putText(frame, scale_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Write frame
    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_idx += 1

    # Progress reporting
    if frame_idx % 60 == 0:
        print(f"📊 Frame {frame_idx}/{total_frames} - 🔴Running: {running_count} | 🟢Not Running: {not_running_count}")

writer.close()

# Detailed final report
print(f"\n🎉 ADAPTIVE Analysis Complete! Output saved to: {output_path}")
print(f"\n📈 DETAILED ADAPTIVE RESULTS:")
print("=" * 100)

for track_id, person in persons.items():
    if person.analysis_complete:
        print(f"\n👤 Person {track_id}: {person.final_classification}")
        print(f"   🎯 Confidence: {person.confidence:.3f}")
        
        if person.gait_metrics_history:
            metrics = person.gait_metrics_history[-1]
            print(f"   📊 Velocity Analysis:")
            print(f"      • Movement Velocity: {metrics.movement_velocity:.1f} px/s")
            print(f"      • Smooth Velocity: {metrics.movement_velocity_smooth:.1f} px/s")
            print(f"      • Peak Velocity: {metrics.movement_velocity_peak:.1f} px/s")
            print(f"      • Walking Threshold: {metrics.adaptive_walking_threshold:.1f} px/s")
            print(f"      • Running Threshold: {metrics.adaptive_running_threshold:.1f} px/s")
            
            print(f"   📊 Other Metrics:")
            print(f"      • Stride Frequency: {metrics.stride_frequency:.2f} Hz")
            print(f"      • Rhythm Consistency: {metrics.rhythm_consistency:.3f}")
            print(f"      • Vertical Oscillation: {metrics.vertical_oscillation:.3f}")
            print(f"      • Ground Contact: {metrics.ground_contact_time:.3f}")
            
            print(f"   👤 Person Characteristics:")
            print(f"      • Estimated Height: {person.estimated_height:.0f} px")
            print(f"      • Personal Scale: {person.personal_velocity_scale:.2f}")
            print(f"      • Visibility: {metrics.partial_visibility_ratio:.1%}")
            
            # Velocity debug information
            if metrics.velocity_debug:
                debug = metrics.velocity_debug
                print(f"   🔍 Velocity Debug:")
                print(f"      • Position Count: {debug.get('position_count', 0)}")
                print(f"      • Velocity Mean: {debug.get('velocity_mean', 0):.1f} px/s")
                print(f"      • Velocity 75th: {debug.get('velocity_75th', 0):.1f} px/s")
                print(f"      • Velocity Max: {debug.get('velocity_max', 0):.1f} px/s")
    else:
        print(f"\n👤 Person {track_id}: ⚠️  INSUFFICIENT DATA ({len(person.keypoint_history)}/{MIN_DETECTION_FRAMES} frames)")


