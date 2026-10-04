"""Visibility gating, smoothing and input-sanitising behaviour of MimicPipeline."""

import math

import numpy as np
import pytest

from filters import OneEuroFilter, SegmentGate
from mediapipe_to_gmr import L_ELBOW, L_WRIST, Segments, neutral_segments
from synthetic_poses import base_poses

BY_NAME = {p.name: p for p in base_poses()}


def _with_visibility(landmarks, indices, visibility):
    arr = landmarks.copy()
    arr[list(indices), 3] = visibility
    return arr


def test_invisible_forearm_holds_then_eases_to_neutral(pipeline):
    """Left wrist goes off-screen while MediaPipe keeps guessing it: the
    left forearm must hold its last seen direction, then ease to hanging,
    and never follow the guessed wrist. The right arm keeps tracking."""
    pipeline.reset()
    goal = BY_NAME["goalpost"].landmarks          # forearms up
    t = 0.0
    for _ in range(10):
        r = pipeline.step(goal, t)
        t += 1 / 30
    held = r.info["segments"].fore["left"].copy()
    assert held[2] > 0.9  # left forearm up

    # MediaPipe's guess for the hidden wrist: pointing forward. Right arm drops.
    guess = BY_NAME["elbows_90_forearms_forward"].landmarks.copy()
    guess = _with_visibility(guess, [L_WRIST], 0.05)
    statuses = []
    while t < 0.33 + pipeline.gate.hold_s + pipeline.gate.ease_s + 0.3:
        r = pipeline.step(guess, t)
        statuses.append(r.status["left_fore"])
        fore = r.info["segments"].fore["left"]
        # Never toward the guessed (forward) direction beyond what easing to
        # "down" passes through: the x (forward) component stays ~0.
        assert abs(fore[0]) < 0.05, fore
        if r.status["left_fore"] == "hold":
            assert np.allclose(fore, held, atol=1e-9)
        t += 1 / 30
    assert statuses[0] == "hold"
    assert "ease" in statuses
    assert statuses[-1] == "neutral"
    assert r.info["segments"].fore["left"][2] < -0.99       # eased to hanging
    assert r.status["right_fore"] == "live"
    assert r.info["segments"].fore["right"][0] > 0.9         # right forearm tracked forward


def test_visibility_hysteresis_does_not_flicker(pipeline):
    pipeline.reset()
    base = BY_NAME["t_pose"].landmarks
    t = 0.0
    seen = []
    for k in range(20):
        vis = 0.5 + (0.03 if k % 2 else -0.03)   # hovering around the 0.5 threshold
        r = pipeline.step(_with_visibility(base, [L_ELBOW], vis), t)
        seen.append(r.status["left_upper"])
        t += 1 / 30
    # Entered below the high threshold -> never valid; must not toggle each frame.
    changes = sum(a != b for a, b in zip(seen, seen[1:]))
    assert changes <= 2, seen


def test_never_seen_segment_starts_neutral():
    gate = SegmentGate()
    seg = Segments()
    seg.fore["left"] = np.array([1.0, 0, 0])
    seg.valid = {k: False for k in seg.valid}
    out, status = gate.update(seg, 0.0)
    assert status["left_fore"] == "neutral"
    assert np.allclose(out.fore["left"], neutral_segments().fore["left"])


def test_one_euro_does_not_overshoot_a_step():
    f = OneEuroFilter()
    x0, x1 = np.zeros((33, 3)), np.ones((33, 3)) * 0.4
    f(x0, 0.0)
    prev = x0
    for k in range(1, 120):
        y = f(x1, k / 30)
        assert np.all(y >= prev - 1e-12)          # monotonic toward the step
        assert np.all(y <= x1 + 1e-12)            # never past it
        prev = y
    assert np.allclose(prev, x1, atol=1e-3)       # and gets there


def test_one_euro_lag_is_small_for_fast_motion():
    """A 1 m/s hand movement should be followed within a few cm."""
    f = OneEuroFilter()
    lag = []
    for k in range(60):
        t = k / 30
        x = np.array([[t * 1.0, 0.0, 0.0]])
        lag.append(abs(f(x, t)[0, 0] - x[0, 0]))
    assert max(lag[30:]) < 0.05


def test_one_euro_reduces_jitter_when_still():
    rng = np.random.default_rng(0)
    f = OneEuroFilter()
    noisy = rng.normal(0, 0.01, size=(300, 33, 3))
    out = np.array([f(n, k / 30) for k, n in enumerate(noisy)])
    assert out[60:].std() < 0.35 * noisy[60:].std()


@pytest.mark.parametrize("corrupt", ["nan_wrist", "inf_all", "zeros", "collapsed_shoulders"])
def test_no_nan_or_inf_reaches_the_controller(pipeline, corrupt):
    pipeline.reset()
    good = BY_NAME["wave"].landmarks
    t = 0.0
    for _ in range(3):
        pipeline.step(good, t)
        t += 1 / 30
    bad = good.copy()
    if corrupt == "nan_wrist":
        bad[L_WRIST, :3] = np.nan
    elif corrupt == "inf_all":
        bad[:, :3] = np.inf
    elif corrupt == "zeros":
        bad[:, :3] = 0.0
    elif corrupt == "collapsed_shoulders":
        bad[12, :3] = bad[11, :3]
    for _ in range(5):
        r = pipeline.step(bad, t)
        t += 1 / 30
        assert all(math.isfinite(v) for v in r.targets.values())
        assert all(math.isfinite(v) for v in r.q.values())
    # And recovers once the input is sane again.
    for _ in range(10):
        r = pipeline.step(good, t)
        t += 1 / 30
    assert all(s == "live" for s in r.status.values())
