import supervision as sv
import os
import logging
import cv2
import numpy as np
from FrameProcessor import FrameProcessor
from ModelLoader import ModelLoader
from Annotator import Annotator

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class MainApplication:
    def __init__(self, model_path: str, selected_classes: list, video_path: str = None, line_zones: list = [((0, 739), (1920, 739))], polygons=[]):
        """Initialize the main application with model, video setup, dynamic line zones, and polygon zones."""
        try:
            self.model_loader = ModelLoader(model_path)
            self.model = self.model_loader.load_model()
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}", exc_info=True)
            raise SystemExit(e)

        self.class_names_dict = self.model.model.names
        self.selected_classes = selected_classes
        self.frame_processor = FrameProcessor(self.model, self.selected_classes)
        self.annotator = Annotator()
        frame_resolution = (1920, 1080)  # Assuming a fixed frame resolution
        for polygon in polygons:
            self.annotator.add_polygon_zone_annotator(polygon, frame_resolution)

        self.setup_video_paths(video_path)

        self.byte_tracker = None  # Initialized later with actual video frame rate
        self.line_zones = [sv.LineZone(start=sv.Point(*start), end=sv.Point(*end)) for start, end in line_zones]
        logging.info("Tracking and multiple line zones setup completed.")

    def setup_video_paths(self, video_path: str = None):
        """Set up source and target video paths dynamically."""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.source_video_path = video_path if video_path else os.path.join(base_dir, "default-video.mp4")
            self.target_video_path = os.path.join(base_dir, "vehicle-counting-output.mp4")
            logging.info(f"Video paths set up successfully: {self.source_video_path}")
        except FileNotFoundError:
            logging.error("Video file not found.")
            raise
        except Exception as e:
            logging.error(f"Error during video setup: {e}", exc_info=True)
            raise

    def run(self):
        """Run the main application to process and annotate video frames."""
        cap = cv2.VideoCapture(self.source_video_path)
        if not cap.isOpened():
            logging.error("Error opening video file.")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        self.byte_tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=frame_rate)
        out = cv2.VideoWriter(self.target_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

        index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.info("End of video file reached.")
                break

            annotated_frame = self.callback(frame, index)
            cv2.putText(annotated_frame, "Sensor Simulation - Abuja Airport Road - ausman", (150, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
            out.write(annotated_frame)
            index += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def callback(self, frame, index: int) -> np.ndarray:
        """Process and annotate a single frame with multiple line zones."""
        try:
            detections = self.frame_processor.process_frame(frame)
            detections = self.byte_tracker.update_with_detections(detections)
            vehicle_detections = detections[np.isin(detections.class_id, [2, 3, 5, 7])]  # Class IDs for motorcycle, bus, truck

            # Update polygon zones with current detections
            for annotator in self.annotator.polygon_zone_annotators:
                annotator.zone.trigger(vehicle_detections)

            labels = [
                f"#{tracker_id} {self.class_names_dict[class_id]} {confidence:.2f}"
                for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence)
            ]
            logging.debug(f"Labels generated for frame {index}: {labels}")
            annotated_frame = self.annotator.annotate_frame(frame.copy(), detections, labels)
            for line_zone in self.line_zones:
                line_zone.trigger(vehicle_detections)
                annotated_frame = self.annotator.line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
            logging.info(f"Frame {index} processed successfully.")
            return annotated_frame
        except Exception as e:
            logging.error(f"Error processing frame {index}: {e}", exc_info=True)
            return frame  # Return unmodified frame on error



if __name__ == "__main__":
    selected_classes = [0, 2, 3, 5, 7]  # car, motorcycle, bus, truck
    video_path = "CHRS9691.mp4"
    line_zones = [((0, 780), (1920, 780)),((168, 717), (181,556)), ((1667, 700), (1667,535))]
    polygons = [
        np.array([[27, 879],[637, 906],[815, 612],[570, 600],[27, 879]]),
        np.array([[1249, 930],[1852, 909],[1318, 603],[1070, 609],[1249, 930]])
    ]
    app = MainApplication("yolov8x.pt", selected_classes, video_path, line_zones, polygons)
    app.run()
