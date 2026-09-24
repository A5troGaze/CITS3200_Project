import cv2

from abstract_camera import AbstractCameraSource


class WebcamSource(AbstractCameraSource):
    """Backend for testing on a laptop: reads from a regular USB/built-in
    webcam via OpenCV."""

    def __init__(self, index=0):
        self.index = index
        self.cap = None

    def init(self):
        self.cap = cv2.VideoCapture(self.index)

    def read(self):
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
