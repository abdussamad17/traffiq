import logging
import supervision as sv
import numpy as np

class Annotator:
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=70)
        self.line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=1)
        self.mask_annotator = sv.MaskAnnotator()
        self.polygon_zone_annotators = []  # List to hold polygon zone annotators
        logging.debug("Annotators initialized.")

    def add_polygon_zone_annotator(self, polygon, frame_resolution):
        """Add a polygon zone annotator for a specific zone."""
        zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=frame_resolution)
        annotator = sv.PolygonZoneAnnotator(
            zone=zone,
            color=sv.Color(30, 144, 255),  # Red color for visibility
            thickness=2,
            text_color=sv.Color(255, 255, 255),  # White text
            text_scale=1,
            text_thickness=2,
            text_padding=10
        )
        self.polygon_zone_annotators.append(annotator)
        logging.debug("Polygon zone annotator added.")

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections, labels: list) -> np.ndarray:
        """Annotate the frame with detection boxes, traces, masks, and polygon zones."""
        logging.debug("Starting frame annotation.")
        frame = self.trace_annotator.annotate(scene=frame, detections=detections)
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        frame = self.mask_annotator.annotate(scene=frame, detections=detections)
        for annotator in self.polygon_zone_annotators:
            frame = annotator.annotate(scene=frame)
        logging.debug("Frame annotation completed.")
        return frame
