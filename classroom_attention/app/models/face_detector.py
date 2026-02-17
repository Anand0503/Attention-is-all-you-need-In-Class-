import mediapipe as mp


class FaceDetector:

    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

    def detect(self, frame):
        rgb = frame[:, :, ::-1]
        results = self.detector.process(rgb)

        boxes = []

        if results.detections:
            h, w, _ = frame.shape

            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                boxes.append([x1, y1, x1 + width, y1 + height])

        return boxes