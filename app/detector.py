import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import numpy as np

class FaceDetector:
    def __init__(self, model_path: str = "detector.tflite") -> None:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, tuple[int, int, int, int]]]]:
        """Detects faces and draws bounding boxes."""
        modified_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=modified_frame)
        result = self.detector.detect(mp_image)
        faces: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
        if result.detections:
            for face in result.detections:
                bbox = face.bounding_box
                x_min = bbox.origin_x
                y_min = bbox.origin_y
                x_max = x_min + bbox.width
                y_max = y_min + bbox.height
                cv.rectangle(modified_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), thickness=2)
                faces.append((frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)))
        final_frame = cv.cvtColor(modified_frame, cv.COLOR_RGB2BGR)
        return final_frame, faces