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

    frame_shown, faces = detector.detect(frame)
    #for face, _ in faces:
        #emotion, confidence = classifier.predict(face)
        #print(emotion, confidence)
    camera.show_image(frame_shown)
       
    if camera.is_exit_required():
        break

camera.release()
