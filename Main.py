import supervision as sv
import os
import logging
import cv2
from FrameProcessor import FrameProcessor
from ModelLoader import ModelLoader
from Annotator import Annotator

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class MainApplication:
    def __init__(self, model_path, selected_classes):
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

        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.source_video_path = os.path.join(base_dir, "VIDEO-2024-04-28-18-38-40.mp4")
            self.target_video_path = os.path.join(base_dir, "vehicle-counting-output.mp4")
            logging.info("Video paths set up successfully.")
        except FileNotFoundError:
            logging.error("Video file not found.")
            raise
        except Exception as e:
            logging.error(f"Error during video setup: {e}", exc_info=True)
            raise

        self.byte_tracker = None  # Initialized later with actual video frame rate
        self.line_zone = sv.LineZone(start=sv.Point(55, 445), end=sv.Point(1171, 494))
        logging.info("Tracking and line zone setup completed.")

    def run(self):
        cap = cv2.VideoCapture(self.source_video_path)  # Open the video file
        if not cap.isOpened():
            logging.error("Error opening video file.")
            return

        # Get video properties to write the output file
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Dynamically get the frame rate

        # Initialize ByteTracker with actual frame rate
        self.byte_tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=frame_rate)

        out = cv2.VideoWriter(self.target_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

        index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.info("End of video file reached.")
                break

            # Process the frame
            annotated_frame = self.callback(frame, index)

            # Add title text to the top right of the frame
            cv2.putText(annotated_frame, "ausman - Sensor simulation", (400, 469), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            # Write the processed frame to the output video
            out.write(annotated_frame)
            index += 1

        # Release everything when job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def callback(self, frame, index):
        try:
            detections = self.frame_processor.process_frame(frame)
            detections = self.byte_tracker.update_with_detections(detections)

            # Generate labels correctly using properties of the 'Detections' object
            labels = [
                f"#{tracker_id} {self.class_names_dict[class_id]} {confidence:.2f}"
                for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence)
            ]
            logging.debug(f"Labels generated for frame {index}: {labels}")
            annotated_frame = self.annotator.annotate_frame(frame.copy(), detections, labels)
            self.line_zone.trigger(detections)
            annotated_frame = self.annotator.line_zone_annotator.annotate(annotated_frame, line_counter=self.line_zone)
            logging.info(f"Frame {index} processed successfully.")
            return annotated_frame
        except Exception as e:
            logging.error(f"Error processing frame {index}: {e}", exc_info=True)
            return frame  # Return unmodified frame on error



if __name__ == "__main__":
    selected_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    app = MainApplication("yolov8x.pt", selected_classes)
    app.run()
