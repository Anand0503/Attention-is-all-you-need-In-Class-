import cv2
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import csv

from app.models.tracker import PersonTracker
from app.models.pose_analyzer import PoseAnalyzer
from app.models.face_detector import FaceDetector


class ClassroomProcessor:

    def __init__(self):
        self.face_detector = FaceDetector()
        self.tracker = PersonTracker()
        self.pose_analyzer = PoseAnalyzer()

        self.student_stats = defaultdict(lambda: {
            "total": 0,
            "attentive": 0,
            "handraise": 0,
            "distracted": 0
        })

    def process_video(self, video_path, output_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error opening video")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, first_frame = cap.read()
        if not ret:
            print("Error reading first frame")
            return

        h, w = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        FRAME_SKIP = 2
        frame_count = 0

        print(f"Total frames: {total_frames}")
        print("Processing video...\n")

        with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if frame_count % FRAME_SKIP != 0:
                    pbar.update(1)
                    continue

                # FACE DETECTION
                face_boxes = self.face_detector.detect(frame)

                detections = []

                for (x1, y1, x2, y2) in face_boxes:
                    width = x2 - x1
                    height = y2 - y1

                    detections.append(
                        ([float(x1), float(y1), float(width), float(height)],
                         1.0,
                         "face")
                    )

                tracks = self.tracker.update(detections, frame)

                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id
                    l, t, r, b = track.to_ltrb()

                    # Pose Analysis
                    if frame_count % 5 == 0:
                        hand_raised, head_forward = self.pose_analyzer.analyze(
                            frame, [l, t, r, b]
                        )
                        track.pose_cache = (hand_raised, head_forward)
                    else:
                        if hasattr(track, "pose_cache"):
                            hand_raised, head_forward = track.pose_cache
                        else:
                            hand_raised, head_forward = (False, False)

                    if hand_raised:
                        status = "Hand Raised"
                        color = (255, 255, 0)
                        self.student_stats[track_id]["handraise"] += 1
                    elif head_forward:
                        status = "Attentive"
                        color = (0, 255, 0)
                        self.student_stats[track_id]["attentive"] += 1
                    else:
                        status = "Distracted"
                        color = (0, 0, 255)
                        self.student_stats[track_id]["distracted"] += 1

                    self.student_stats[track_id]["total"] += 1

                    cv2.rectangle(frame,
                                  (int(l), int(t)),
                                  (int(r), int(b)),
                                  color, 2)

                    cv2.putText(frame,
                                f"ID {track_id} {status}",
                                (int(l), int(t) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2)

                out.write(frame)
                pbar.update(1)

        cap.release()
        out.release()

        print("\nProcessing completed.")
        self.save_results("outputs/attention_results.csv")

    def save_results(self, path):

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "Student_ID",
                "Total_Frames",
                "Attentive_Frames",
                "HandRaise_Frames",
                "Distracted_Frames",
                "Attention_Percentage"
            ])

            for sid, stats in self.student_stats.items():

                if stats["total"] < 30:
                    continue

                total = stats["total"]
                attentive = stats["attentive"]

                percentage = round((attentive / total) * 100, 2)

                writer.writerow([
                    sid,
                    total,
                    attentive,
                    stats["handraise"],
                    stats["distracted"],
                    percentage
                ])