"""
Confidence-Weighted Attention Engine.

Computes attention scores per student using detection and pose confidences
as weights, with rolling window smoothing. Tracks per-student frame counters
for attentive, distracted, hand-raise, and phone-usage states.
"""

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class AttentionEngine:
    """
    Confidence-weighted attention scoring with rolling window smoothing.

    Scoring formula:
        - Looking forward:  +weight × pose_confidence
        - Hand raised:      +weight × detection_confidence
        - Distracted:       +weight × pose_confidence  (negative weight)
        - Using phone:      +weight × detection_confidence (negative weight)

    Attributes:
        smoothing_window: Number of recent scores to average.
        confidence_weighted: Whether to use confidence weighting.
        weights: Dict of behavior weights.
        history: Per-student rolling score history.
        counters: Per-student frame counters.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize AttentionEngine.

        Args:
            config: Full pipeline config dict. Uses 'attention' section.
        """
        config = config or {}
        att_cfg = config.get("attention", {})

        self.smoothing_window = att_cfg.get("smoothing_window", 10)
        self.confidence_weighted = att_cfg.get("confidence_weighted", True)

        default_weights = {
            "looking_forward": 1.0,
            "hand_raised": 1.0,
            "distracted": -1.0,
            "using_phone": -2.0,
        }
        self.weights = att_cfg.get("weights", default_weights)

        # Per-student score history for smoothing
        self.history: dict[str, list[float]] = defaultdict(list)

        # Per-student frame counters
        self.counters: dict[str, dict] = defaultdict(lambda: {
            "attentive_frames": 0,
            "distracted_frames": 0,
            "handraise_frames": 0,
            "using_phone_frames": 0,
            "total_frames": 0,
            "cumulative_weighted_score": 0.0,
        })

    def update(self, person_id: str, signals: dict) -> str:
        """
        Update attention score for a person based on behavior signals.

        Args:
            person_id: Global student ID.
            signals: Dict with keys:
                'hand_raised': bool
                'head_forward': bool
                'using_phone': bool
                'pose_confidence': float (0-1)
                'detection_confidence': float (0-1)

        Returns:
            Attention status string: 'attentive', 'distracted', or 'neutral'.
        """
        hand_raised = signals.get("hand_raised", False)
        head_forward = signals.get("head_forward", False)
        using_phone = signals.get("using_phone", False)
        pose_conf = signals.get("pose_confidence", 0.5)
        det_conf = signals.get("detection_confidence", 0.5)

        score = 0.0
        counters = self.counters[person_id]
        counters["total_frames"] += 1

        if self.confidence_weighted:
            # Confidence-weighted scoring
            if hand_raised:
                score += self.weights["hand_raised"] * det_conf
                counters["handraise_frames"] += 1
            if head_forward and not using_phone:
                score += self.weights["looking_forward"] * pose_conf
                counters["attentive_frames"] += 1
            if using_phone:
                score += self.weights["using_phone"] * det_conf
                counters["using_phone_frames"] += 1
            if not head_forward and not hand_raised and not using_phone:
                score += self.weights["distracted"] * pose_conf
                counters["distracted_frames"] += 1
        else:
            # Simple binary scoring (legacy fallback)
            if hand_raised:
                score += 2.0
                counters["handraise_frames"] += 1
            if head_forward:
                score += 1.0
                counters["attentive_frames"] += 1
            if using_phone:
                score -= 2.0
                counters["using_phone_frames"] += 1
            if not head_forward and not hand_raised:
                score -= 1.0
                counters["distracted_frames"] += 1

        counters["cumulative_weighted_score"] += score

        # Rolling window smoothing
        self.history[person_id].append(score)
        if len(self.history[person_id]) > self.smoothing_window:
            self.history[person_id] = self.history[person_id][
                -self.smoothing_window:
            ]

        avg_score = float(np.mean(self.history[person_id]))

        if avg_score > 0.3:
            return "attentive"
        elif avg_score < -0.3:
            return "distracted"
        else:
            return "neutral"

    def get_student_metrics(self, person_id: str) -> dict:
        """
        Get comprehensive attention metrics for a student.

        Args:
            person_id: Global student ID.

        Returns:
            Dict with all counters, weighted score, and attention percentage.
        """
        counters = self.counters.get(person_id, {
            "attentive_frames": 0,
            "distracted_frames": 0,
            "handraise_frames": 0,
            "using_phone_frames": 0,
            "total_frames": 0,
            "cumulative_weighted_score": 0.0,
        })

        total = counters["total_frames"]
        attentive = counters["attentive_frames"]
        attention_pct = round((attentive / total) * 100, 2) if total > 0 else 0.0

        return {
            "attentive_frames": attentive,
            "distracted_frames": counters["distracted_frames"],
            "handraise_frames": counters["handraise_frames"],
            "using_phone_frames": counters["using_phone_frames"],
            "total_frames": total,
            "confidence_weighted_score": round(
                counters["cumulative_weighted_score"], 4
            ),
            "attention_percentage": attention_pct,
        }

    def get_all_metrics(self) -> dict[str, dict]:
        """Get metrics for all tracked students."""
        return {pid: self.get_student_metrics(pid) for pid in self.counters}

    def reset(self):
        """Reset all histories and counters."""
        self.history.clear()
        self.counters.clear()
        logger.info("AttentionEngine state reset.")