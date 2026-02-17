from deep_sort_realtime.deepsort_tracker import DeepSort


class PersonTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=120,
            n_init=3,
            max_cosine_distance=0.3,
            nn_budget=None
        )

    def update(self, detections, frame):
        return self.tracker.update_tracks(detections, frame=frame)