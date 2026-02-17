import mediapipe as mp


class PoseAnalyzer:

    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False
        )

    def analyze(self, frame, bbox):
        l, t, r, b = map(int, bbox)

        # Clamp coordinates
        l = max(0, l)
        t = max(0, t)
        r = min(frame.shape[1], r)
        b = min(frame.shape[0], b)

        person_crop = frame[t:b, l:r]
        if person_crop.size == 0:
            return False, False

        rgb = person_crop[:, :, ::-1]
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return False, False

        landmarks = results.pose_landmarks.landmark

        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        nose = landmarks[0]

        # Hand raised
        hand_raised = (
            left_wrist.y < left_shoulder.y or
            right_wrist.y < right_shoulder.y
        )

        # Head forward
        head_forward = abs(nose.x - 0.5) < 0.2

        return hand_raised, head_forward