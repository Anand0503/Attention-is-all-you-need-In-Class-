# Attention-is-all-you-need-In-Class-

## 🎓 Classroom Attention Monitoring System

A Computer Vision-based system for analyzing student attention in classroom videos using face detection, tracking, and head pose estimation.

This project processes a classroom video, tracks students, evaluates attention behavior, and generates per-student analytics in CSV format.

---

## 🚀 Features

- ✅ Face-based student tracking
- ✅ Stable tracking IDs within a video
- ✅ Head pose–based attention detection
- ✅ Optional hand raise detection
- ✅ Rolling smoothing for stable classification
- ✅ Per-student statistics collection
- ✅ Automatic CSV report generation
- ✅ CPU compatible
- ✅ tqdm progress bar support

---

## 🏗 Project Structure

```
classroom_attention/
│
├── app/
│ ├── models/
│ │ ├── face_detector.py
│ │ ├── tracker.py
│ │ ├── pose_analyzer.py
│ │ └── detectors.py (if using YOLO)
│ │
│ └── pipeline/
│ └── processor.py
│
├── outputs/
│ ├── output.avi
│ └── attention_results.csv
│
├── run.py
├── requirements.txt
└── README.md
```

---

## 🧠 System Architecture

```
Video Input
↓
Face Detection (MediaPipe)
↓
DeepSORT Tracking
↓
Head Pose Analysis
↓
Attention Scoring
↓
Per-Student Statistics
↓
CSV Report Generation
```

---

## 📦 Installation

### 1️⃣ Clone the Repository
```
git clone <your-repository-url>
cd classroom_attention
```
### 2️⃣ Create Virtual Environment
```
python3 -m venv venv
source venv/bin/activate
```
### 3️⃣ Install Dependencies
```pip install -r requirements.txt```

Or manually:

```pip install ultralytics deep-sort-realtime mediapipe tqdm numpy opencv-python```
### ▶️ Running the Project

Place your classroom video in the root folder as:

```input.mp4```

Run the system:

```python run.py```

### 📤 Output Files

After processing completes:
```
outputs/output.avi
outputs/attention_results.csv
```
### 📊 CSV Output Format

The CSV file contains per-student analytics:
```
Student_ID,
Total_Frames,
Attentive_Frames,
HandRaise_Frames,
Distracted_Frames,
Attention_Percentage
```
#### Example Output
```
2,252,110,114,28,43.65
3,3057,195,2622,240,6.38
```

## 📈 Attention Calculation Logic

For each tracked student:

### +1 → Looking forward

### +1 → Hand raised (if enabled)

### -2 → Using phone (if enabled)

Final attention percentage:

```
(attentive_frames / total_frames) × 100
```

Short-lived tracks are filtered to avoid false student counts.

## ⚡ Performance Notes

- Pose estimation runs every 5 frames (CPU optimization)

- DeepSORT parameters tuned for classroom stability

- Face detection reduces false positives

- Tracking IDs are stable within a single video

---

## 🛠 Technologies Used

- Python 3.10

- OpenCV

- MediaPipe

- DeepSORT

- YOLO (optional)

- NumPy

- tqdm

---
🔮 Future Improvements

Real-time webcam version

Web dashboard for visualization

Face recognition for persistent IDs across sessions

FastAPI deployment

Docker containerization

Classroom-level attention analytics dashboard

---

📜 License

This project is developed for academic and research purposes.


---

