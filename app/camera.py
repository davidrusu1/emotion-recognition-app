#!/usr/bin/env python3
import cv2 as cv
import numpy as np

class Camera:
    def __init__(self) -> None:
        self.camera = cv.VideoCapture(0)

    def release(self) -> None:
        self.camera.release()
        cv.destroyAllWindows()

    def read(self) -> np.ndarray | None:
        """Reads one frame from the webcam."""
        ret, frame = self.camera.read()
        if not ret:
            return None
        return frame

    def show_image(self, frame) -> None:
        frame = cv.flip(frame, 1)
        cv.imshow('Camera', frame)

    def is_exit_required(self) -> bool:
        """Checks if q is pressed then it will close the app"""
        if cv.waitKey(1) & 0xFF == ord('q'):
            return True
        return False
    