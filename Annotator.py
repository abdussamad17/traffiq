import logging
import supervision as sv

class Annotator:
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)
        self.line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)
        self.mask_annotator = sv.MaskAnnotator()
        logging.debug("Annotators initialized.")

    def annotate_frame(self, frame, detections, labels):
        logging.debug("Starting frame annotation.")
        frame = self.trace_annotator.annotate(scene=frame, detections=detections)
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        frame = self.mask_annotator.annotate(scene=frame, detections=detections)
        logging.debug("Frame annotation completed.")
        return frame
