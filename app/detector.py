import mediapipe as mp
import cv2 as cv
import numpy as np

class FaceDetector:
    def __init__(self) -> None:
        self.detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence = 0.5)

    def detect(self, frame) -> np.ndarray:
        """Detects faces and draws bounding boxes."""
        modified_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.detector.process(modified_frame)
        if results.detections:
            for face in results.detections:
                bbox = face.location_data.relative_bounding_box
                image_height, image_width, _ = modified_frame.shape
                x_min = int(bbox.xmin * image_width)
                y_min = int(bbox.ymin * image_height)
                x_max = int(x_min + (bbox.width * image_width))
                y_max = int(y_min + (bbox.height * image_height))
                cv.rectangle(modified_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        final_frame = cv.cvtColor(modified_frame, cv.COLOR_RGB2BGR)
        return final_frame