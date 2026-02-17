from collections import defaultdict
import numpy as np

class AttentionEngine:
    def __init__(self):
        self.history = defaultdict(list)

    def update(self, person_id, signals):
        score = 0

        if signals.get("hand_raise"):
            score += 2
        if signals.get("note_taking"):
            score += 1.5
        if signals.get("looking_forward"):
            score += 1
        if signals.get("phone"):
            score -= 2
        if signals.get("bow_head"):
            score -= 2.5

        self.history[person_id].append(score)

        # keep last 30 frames
        if len(self.history[person_id]) > 30:
            self.history[person_id] = self.history[person_id][-30:]

        avg_score = np.mean(self.history[person_id])

        if avg_score > 0.7:
            return "attentive"
        elif avg_score < -0.3:
            return "inattentive"
        else:
            return "neutral"