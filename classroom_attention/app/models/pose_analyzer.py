"""
Pose analysis module using YOLO Pose model (MediaPipe-free).

Uses a YOLO pose estimation model to detect body keypoints and classify:
    - Hand raised (wrist above shoulder)
    - Head facing forward (nose centered)
    - Phone usage heuristic (wrists near face, head down)
    - Pose detection confidence

Returns structured results with confidence scores for
confidence-weighted attention scoring.
"""

import logging
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class PoseAnalyzer:
    """
    Analyzes body pose using YOLO pose estimation to determine
    student behavior signals.

    YOLO Pose keypoints (COCO format):
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip, ...

    Attributes:
        model: YOLO pose model instance.
        device: Compute device.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize PoseAnalyzer with YOLO pose model.

        Args:
            config: Full pipeline config dict. Uses 'system' section for device.
        """
        config = config or {}
        sys_cfg = config.get("system", {})

        # Auto-select device
        device = sys_cfg.get("device", "auto")
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        # Use YOLO pose model — lightweight nano variant
        model_name = "yolo11n-pose.pt"
        logger.info(f"Loading PoseAnalyzer model: {model_name} on {self.device}")
        self.model = YOLO(model_name)

    def analyze(self, frame, bbox):
        """
        Analyze pose for a single person crop.

        Args:
            frame: Full BGR frame (numpy array, H x W x 3).
            bbox: Bounding box as [left, top, right, bottom].

        Returns:
            dict with keys:
                'hand_raised': bool
                'head_forward': bool
                'using_phone': bool
                'pose_confidence': float (0.0–1.0)
        """
        default_result = {
            "hand_raised": False,
            "head_forward": False,
            "using_phone": False,
            "pose_confidence": 0.0,
        }

        try:
            l, t, r, b = map(int, bbox)

            # Clamp to frame bounds
            h_frame, w_frame = frame.shape[:2]
            l = max(0, l)
            t = max(0, t)
            r = min(w_frame, r)
            b = min(h_frame, b)

            if r <= l or b <= t:
                return default_result

            person_crop = frame[t:b, l:r]
            if person_crop.size == 0:
                return default_result

            # Run YOLO pose on the person crop
            results = self.model(
                person_crop,
                device=self.device,
                verbose=False,
            )[0]

            if results.keypoints is None or len(results.keypoints) == 0:
                return default_result

            # Get first person's keypoints (should be the main person in crop)
            kpts = results.keypoints[0]

            # keypoints.data shape: (num_people, 17, 3) where 3 = x, y, conf
            if kpts.data is None or kpts.data.shape[-1] < 3:
                return default_result

            kp_data = kpts.data[0]  # (17, 3) — first person
            crop_h, crop_w = person_crop.shape[:2]

            # Normalize coordinates to 0-1 range
            kp_norm = kp_data.clone()
            kp_norm[:, 0] = kp_norm[:, 0] / crop_w  # x
            kp_norm[:, 1] = kp_norm[:, 1] / crop_h  # y

            # Extract key landmarks (COCO keypoint indices)
            nose = kp_norm[0]          # (x, y, conf)
            l_shoulder = kp_norm[5]
            r_shoulder = kp_norm[6]
            l_elbow = kp_norm[7]
            r_elbow = kp_norm[8]
            l_wrist = kp_norm[9]
            r_wrist = kp_norm[10]

            # --- Pose confidence ---
            key_confs = [
                nose[2].item(), l_shoulder[2].item(), r_shoulder[2].item(),
                l_wrist[2].item(), r_wrist[2].item(),
                l_elbow[2].item(), r_elbow[2].item(),
            ]
            pose_confidence = float(np.mean(key_confs))

            # Filter out low-confidence keypoints
            conf_threshold = 0.3
            nose_valid = nose[2].item() > conf_threshold
            l_shoulder_valid = l_shoulder[2].item() > conf_threshold
            r_shoulder_valid = r_shoulder[2].item() > conf_threshold
            l_wrist_valid = l_wrist[2].item() > conf_threshold
            r_wrist_valid = r_wrist[2].item() > conf_threshold

            # --- Hand raised ---
            hand_raised = False
            if l_wrist_valid and l_shoulder_valid:
                if l_wrist[1].item() < l_shoulder[1].item():
                    hand_raised = True
            if r_wrist_valid and r_shoulder_valid:
                if r_wrist[1].item() < r_shoulder[1].item():
                    hand_raised = True

            # --- Head facing forward ---
            head_forward = False
            if nose_valid:
                # Nose centered horizontally (0.5 = center of crop)
                head_forward = abs(nose[0].item() - 0.5) < 0.2

            # --- Phone usage heuristic ---
            using_phone = False
            mid_shoulder_y = 0.5
            if l_shoulder_valid and r_shoulder_valid:
                mid_shoulder_y = (l_shoulder[1].item() + r_shoulder[1].item()) / 2.0

            if l_wrist_valid and r_wrist_valid and nose_valid:
                wrists_near_face = (
                    l_wrist[1].item() < mid_shoulder_y and
                    r_wrist[1].item() < mid_shoulder_y and
                    l_wrist[1].item() > nose[1].item() - 0.1 and
                    r_wrist[1].item() > nose[1].item() - 0.1
                )
                wrists_close = abs(l_wrist[0].item() - r_wrist[0].item()) < 0.3
                head_down = nose[1].item() > mid_shoulder_y - 0.05

                using_phone = wrists_near_face and wrists_close and head_down

            # If using phone, override head_forward
            if using_phone:
                head_forward = False

            return {
                "hand_raised": hand_raised,
                "head_forward": head_forward,
                "using_phone": using_phone,
                "pose_confidence": pose_confidence,
            }

        except Exception as e:
            logger.warning(f"Pose analysis failed: {e}")
            return default_result

    def close(self):
        """No explicit cleanup needed for YOLO model."""
        pass