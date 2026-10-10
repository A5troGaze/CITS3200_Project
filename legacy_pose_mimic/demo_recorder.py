"""
Side-by-side proof video: left = the annotated camera frame (or, for a
replay, a stick figure of the recorded landmarks), right = an offscreen
MuJoCo render of unitree_mujoco's G1 driven by the same joint targets.

The robot on the right is simulated here, not captured from the DDS sim:
the fixed-base G1 MJCF stepped with the simulator's PD law, the SDK gains
and gravity feed-forward (g1_sim.PdSim), fed through the same
CommandShaper the DDS controller uses. What it shows is therefore what
the simulator does with those targets, frame-synchronised with the video.

Offscreen rendering uses mujoco.Renderer. On a machine without a display
set MUJOCO_GL=egl (or osmesa). The team VM has no GPU (llvmpipe): one
480x480 frame takes ~0.1 s even with multisampling, shadows and
reflections off, so rendering live would drop the camera loop to ~7 fps.
The recorder therefore only stores the camera frames (to a temporary
video) and the per-frame targets while running, and composes the
side-by-side video in close(), after the session.
"""

import os
import tempfile

import numpy as np

from command_shaping import CommandShaper
from g1_sim import PdSim
from pose_common import POSE_CONNECTIONS

PANEL_H = 480
LEFT_SIDE = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}


class RobotView:
    """Fixed-base G1 simulation plus offscreen renderer."""

    def __init__(self, width=PANEL_H, height=PANEL_H, fps=30.0):
        import mujoco

        self._mujoco = mujoco
        self.sim = PdSim()
        self.sim.model.vis.quality.offsamples = 0
        self.steps_per_frame = max(1, int(round((1.0 / fps) / self.sim.model.opt.timestep)))
        self.shaper = CommandShaper(self.sim.q_des.copy(), self.sim.model.opt.timestep)
        self.renderer = mujoco.Renderer(self.sim.model, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (0.0, 0.0, 0.95)
        cam.distance = 2.3
        cam.azimuth = 180.0     # looking at the robot's front (robot faces +x)
        cam.elevation = -8.0
        self.camera = cam

    def advance(self, targets):
        for idx, q in targets.items():
            self.shaper.set_target(idx, q)
        for _ in range(self.steps_per_frame):
            self.sim.set_targets({i: v for i, v in enumerate(self.shaper.tick())})
            self.sim.step(1)

    def render_bgr(self):
        self.renderer.update_scene(self.sim.data, camera=self.camera)
        flags = self.renderer.scene.flags
        for flag in (self._mujoco.mjtRndFlag.mjRND_SHADOW, self._mujoco.mjtRndFlag.mjRND_REFLECTION,
                     self._mujoco.mjtRndFlag.mjRND_SKYBOX):
            flags[flag] = False
        rgb = self.renderer.render()
        return rgb[:, :, ::-1].copy()

    def close(self):
        self.renderer.close()


def stick_figure(landmarks, size=PANEL_H, scale=200.0):
    """Orthographic front view of MediaPipe world landmarks (person facing
    the camera, as the camera sees them). Person's left side in orange."""
    import cv2

    img = np.full((size, size, 3), 30, dtype=np.uint8)
    arr = np.asarray(landmarks, dtype=float)
    pts = [(int(size / 2 + scale * x), int(size * 0.62 + scale * y)) for x, y in arr[:, :2]]
    vis = arr[:, 3] if arr.shape[1] > 3 else np.ones(len(arr))
    for a, b in POSE_CONNECTIONS:
        colour = (0, 165, 255) if (a in LEFT_SIDE and b in LEFT_SIDE) else (255, 200, 80)
        if min(vis[a], vis[b]) < 0.5:
            colour = (90, 90, 90)
        cv2.line(img, pts[a], pts[b], colour, 3)
    for k, p in enumerate(pts):
        cv2.circle(img, p, 4, (255, 255, 255) if vis[k] >= 0.5 else (90, 90, 90), -1)
    cv2.putText(img, "leader (replay)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    return img


def _fit(img, height):
    import cv2

    h, w = img.shape[:2]
    return cv2.resize(img, (int(round(w * height / h)), height))


class DemoRecorder:
    """Collects (left frame, targets) per frame; close() renders the robot
    panel and writes the side-by-side mp4."""

    def __init__(self, path, fps=30.0, height=PANEL_H, log=print):
        self.path = path
        self.fps = float(fps)
        self.height = height
        self.log = log
        self.items = []            # (kind, payload, targets, label)
        self._tmp = None
        self._tmp_writer = None

    @staticmethod
    def _label(result):
        if result is None:
            return "no leader"
        live = sum(1 for s in result.status.values() if s == "live")
        return f"limbs live {live}/5   GMR {result.timings_ms['gmr']:.1f} ms"

    def add_camera_frame(self, frame_bgr, result):
        """Live/video input. result may be None when no leader is tracked
        (the robot then keeps its last targets)."""
        import cv2

        if self._tmp_writer is None:
            fd, self._tmp = tempfile.mkstemp(suffix=".avi", prefix="demo_cam_")
            os.close(fd)
            h, w = frame_bgr.shape[:2]
            self._tmp_writer = cv2.VideoWriter(self._tmp, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h))
        self._tmp_writer.write(frame_bgr)
        self.items.append(("cam", None, None if result is None else dict(result.targets), self._label(result)))

    def add_stick_frame(self, landmarks, result):
        self.items.append(("stick", np.array(landmarks, dtype=float), dict(result.targets), self._label(result)))

    def close(self):
        import cv2

        if self._tmp_writer is not None:
            self._tmp_writer.release()
        if not self.items:
            return
        self.log(f"Rendering demo video ({len(self.items)} frames)...")
        robot = RobotView(width=self.height, height=self.height, fps=self.fps)
        cam = cv2.VideoCapture(self._tmp) if self._tmp else None
        writer = None
        try:
            for k, (kind, payload, targets, label) in enumerate(self.items):
                robot.advance(targets or {})
                if kind == "cam":
                    ok, left = cam.read()
                    if not ok:
                        break
                else:
                    left = stick_figure(payload, self.height)
                right = robot.render_bgr()
                cv2.putText(right, "G1 (unitree_mujoco model)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1)
                cv2.putText(right, label, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                frame = np.hstack([_fit(left, self.height), right])
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                    if not writer.isOpened():
                        raise SystemExit(f"Could not open demo video '{self.path}' for writing")
                writer.write(frame)
                if k and k % 150 == 0:
                    self.log(f"  {k}/{len(self.items)}")
        finally:
            if writer is not None:
                writer.release()
            if cam is not None:
                cam.release()
            if self._tmp and os.path.exists(self._tmp):
                os.remove(self._tmp)
            robot.close()
