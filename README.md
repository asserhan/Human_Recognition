# Human_Recognition

# Advanced Running Detection System

An intelligent computer vision system that automatically detects and classifies human running vs walking using YOLO pose estimation and advanced biomechanical analysis.

## 🎯 Project Overview

This system solves the challenging problem of accurately distinguishing between running and walking in video footage using pose estimation and gait analysis. The algorithm adapts to different video characteristics and provides reliable classification even with partial occlusion or varying camera angles.

## 🚀 Key Features

- **Adaptive Thresholds**: Automatically scales detection parameters based on video resolution and characteristics
- **Multi-Stage Validation**: Uses 3-stage classification system for robust detection
- **Biomechanical Analysis**: Analyzes stride frequency, vertical oscillation, ground contact time, and movement velocity
- **Person-Specific Scaling**: Adapts to individual person size and distance from camera
- **Real-time Processing**: Processes video streams with live classification results
- **Debug Visualization**: Comprehensive debugging information and metrics display

## 🛠️ Installation

Install with:
```
pip install -r requirements.txt
```


## 🎮 Usage

### Basic Usage

1. **Place your video file** in the project directory
2. **Update the video path** in the script:
   ```python
   video_path = "running_person.mp4"
   output_path = "output_results.mp4"
   ```
3. **Run the detection**:
   ```
   python yolov8n_pose.py
    ```
### Configuration Parameters

Key parameters you can adjust in the script:

```python
# Analysis Parameters
ANALYSIS_WINDOW = 60              # Frames to analyze
MIN_DETECTION_FRAMES = 25         # Minimum frames needed for classification

# Adaptive Thresholds (automatically scaled)
BASE_WALKING_VELOCITY = 8.0       # Base walking speed in px/s
BASE_RUNNING_VELOCITY = 15.0      # Base running speed in px/s

# Frequency Thresholds
MIN_STRIDE_FREQUENCY = 1.6        # Minimum stride frequency (Hz)
RUNNING_FREQUENCY_RANGE = (1.8, 4.5)  # Running frequency range (Hz)

# Other Biomechanical Thresholds
MIN_VERTICAL_OSCILLATION = 0.08   # Minimum vertical bounce
MAX_GROUND_CONTACT = 0.60         # Maximum ground contact ratio
MIN_RHYTHM_CONSISTENCY = 0.25     # Minimum rhythm consistency

# Debug Mode
DEBUG_MODE = True                 # Show detailed metrics on video
