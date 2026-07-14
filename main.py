#!/usr/bin/env python3
from app.camera import Camera
from app.detector import FaceDetector

camera = Camera()  # initialise the camera
detector = FaceDetector() # initialise the face detector
while True:
    frame = camera.read()
    if frame is None:
        print("Camera is not connected")
        break

    camera.show_image(detector.detect(frame))
       
    if camera.is_exit_required():
        break

camera.release()
