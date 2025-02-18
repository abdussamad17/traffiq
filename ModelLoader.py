import ultralytics
import logging

class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        logging.debug(f"ModelLoader initialized with model path: {self.model_path}")

    def load_model(self) -> ultralytics.YOLO:
        """Load and optimize the YOLO model from a specified path."""
        logging.info(f"Loading model from {self.model_path}")
        model = ultralytics.YOLO(self.model_path)
        model.fuse()  # Optimize the model
        logging.info("Model loaded and optimized.")
        return model
