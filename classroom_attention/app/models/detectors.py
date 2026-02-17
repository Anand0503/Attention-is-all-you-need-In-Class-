from ultralytics import YOLO



class BehaviorDetector:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        return results


class HandRaiseDetector:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        return results


class PoseDetector:
    def __init__(self, weight_path="yolo11n-pose.pt"):
        self.model = YOLO(weight_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        return results
    
class PersonDetector:
    def __init__(self, weight="yolov8n.pt"):
        self.model = YOLO(weight)

    def detect(self, frame):
        results = self.model(frame, device="cpu", verbose=False)[0]
        return results