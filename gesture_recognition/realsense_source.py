import numpy as np
import pyrealsense2 as rs

from abstract_camera import AbstractCameraSource


class RealsenseSource(AbstractCameraSource):
    """Backend for the real robot: reads the color stream from the G1's
    onboard Intel RealSense D435i via Intel's own pyrealsense2 SDK
    (this is a separate library from unitree_sdk2py/CycloneDDS -- the
    depth camera isn't exposed through Unitree's own DDS messages).

    NOTE: pyrealsense2 doesn't ship official macOS wheels, so this can't
    be pip-installed or tested on a Mac -- it's meant to run on the
    robot's own onboard Linux computer. This has only been reviewed for
    correctness against Intel's documented API, not run against real
    hardware.
    """

    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None

    def init(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.pipeline.start(config)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None
        # bgr8 format matches what cv2/mediapipe already expect elsewhere
        # in this codebase, so no extra color conversion is needed here.
        frame = np.asanyarray(color_frame.get_data())
        return True, frame

    def release(self):
        if self.pipeline is not None:
            self.pipeline.stop()
