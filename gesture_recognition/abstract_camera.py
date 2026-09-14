from abc import ABC, abstractmethod


class AbstractCameraSource(ABC):
    """Shared interface both camera sources must implement, mirroring
    AbstractGestureController's sim/real split. gesture_control.py only
    ever calls these three methods, so swapping the laptop webcam for the
    G1's onboard camera is a one-argument change, not a code change."""

    @abstractmethod
    def init(self):
        """Open/connect to whatever this source reads frames from."""

    @abstractmethod
    def read(self):
        """Return (ok, frame) for the next available frame, same shape as
        cv2.VideoCapture.read(): ok is False when no frame is available."""

    @abstractmethod
    def release(self):
        """Release/close the underlying camera resource."""
