import supervision as sv
import numpy as np
import logging

class FrameProcessor:
    def __init__(self, model, selected_classes):
        self.model = model
        self.selected_classes = selected_classes
        logging.debug("FrameProcessor initialized with selected classes.")

    def process_frame(self, frame):
        logging.debug("Processing frame.")
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        # Filter detections based on selected classes
        detections = detections[np.isin(detections.class_id, self.selected_classes)]
        return detections
