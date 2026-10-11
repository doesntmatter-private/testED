"""Unit tests for the intrusion state machine. Run with: python -m unittest -v  (or pytest)."""
import unittest

from laneguard.geometry import point_in_polygon
from laneguard.intrusion import IntrusionState, LaneConfig

# Closed lane: a trapezoid in a 1920x1080 frame, wide at the bottom (near camera), narrow at the top.
# Crew is at the bottom of the frame, so inbound traffic moves +y.
LANE = [(700, 1080), (1200, 1080), (1000, 500), (900, 500)]
CREW_DIR = (0, 1)
FPS = 30.0


def cfg(**kw):
    return LaneConfig(polygon=LANE, crew_direction=CREW_DIR, **kw)


def box_at(cx, foot_y, w=60, h=40):
    return (cx - w / 2, foot_y - h, cx + w / 2, foot_y)


def drive(state, track_id, xs, ys, t0=0.0):
    """Feed a trajectory one frame at a time, return all events."""
    events = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        t = t0 + i / FPS
        state.update(track_id, box_at(x, y), t)
        state.end_frame(t)
        events += state.pop_events()
    return events


class GeometryTests(unittest.TestCase):
    def test_inside_outside(self):
        self.assertTrue(point_in_polygon((950, 900), LANE))
        self.assertFalse(point_in_polygon((300, 900), LANE))
        self.assertFalse(point_in_polygon((950, 200), LANE))


class IntrusionTests(unittest.TestCase):
    def test_straight_through_alarms_once(self):
        s = IntrusionState(cfg())
        ys = list(range(400, 1080, 20))  # 34 frames, 20 px per frame toward crew
        ev = drive(s, 1, [950] * len(ys), ys)
        alarms = [e for e in ev if e.kind == "alarm"]
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0].track_id, 1)
        # must fire within confirm_frames of first entering the polygon (y > 500)
        first_inside = next(i for i, y in enumerate(ys) if point_in_polygon((950, y), LANE))
        alarm_frame = round(alarms[0].t * FPS)
        self.assertLessEqual(alarm_frame - first_inside, cfg().confirm_frames)

    def test_vehicle_in_open_lane_never_alarms(self):
        s = IntrusionState(cfg())
        ys = list(range(400, 1080, 20))
        ev = drive(s, 2, [300] * len(ys), ys)
        self.assertEqual([e for e in ev if e.kind == "alarm"], [])

    def test_late_merge_is_near_miss_not_alarm(self):
        s = IntrusionState(cfg())
        # 2 frames inside, then merges out to the left and continues in open lane
        xs = [950, 950, 950, 850, 650, 500, 400, 400, 400, 400]
        ys = [480, 500, 520, 540, 560, 580, 600, 620, 640, 660]
        ev = drive(s, 3, xs, ys)
        self.assertEqual([e.kind for e in ev if e.kind == "alarm"], [])
        self.assertEqual(len([e for e in ev if e.kind == "near_miss"]), 1)

    def test_stopped_vehicle_inside_does_not_alarm(self):
        s = IntrusionState(cfg())
        ev = drive(s, 4, [950] * 40, [900] * 40)  # parked work truck
        self.assertEqual(ev, [])

    def test_outbound_vehicle_inside_does_not_alarm(self):
        s = IntrusionState(cfg())
        ys = list(range(1000, 500, -20))  # driving away from crew, up the frame
        ev = drive(s, 5, [950] * len(ys), ys)
        self.assertEqual([e for e in ev if e.kind == "alarm"], [])

    def test_global_cooldown_suppresses_second_alarm_in_platoon(self):
        s = IntrusionState(cfg(cooldown_s=8.0))
        ys = list(range(400, 1080, 20))
        n = len(ys)
        # two vehicles one frame apart
        events = []
        for i in range(n):
            t = i / FPS
            s.update(10, box_at(950, ys[i]), t)
            if i >= 1:
                s.update(11, box_at(950, ys[i - 1]), t)
            s.end_frame(t)
            events += s.pop_events()
        alarms = [e for e in events if e.kind == "alarm"]
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0].track_id, 10)

    def test_second_alarm_allowed_after_cooldown(self):
        s = IntrusionState(cfg(cooldown_s=1.0))
        ys = list(range(400, 1080, 20))
        ev1 = drive(s, 20, [950] * len(ys), ys, t0=0.0)
        ev2 = drive(s, 21, [950] * len(ys), ys, t0=5.0)
        self.assertEqual(len([e for e in ev1 if e.kind == "alarm"]), 1)
        self.assertEqual(len([e for e in ev2 if e.kind == "alarm"]), 1)

    def test_track_expires_after_timeout(self):
        s = IntrusionState(cfg(track_timeout_s=0.5))
        s.update(30, box_at(950, 900), 0.0)
        s.end_frame(0.0)
        self.assertIn(30, s.track_ids())
        s.end_frame(1.0)
        self.assertNotIn(30, s.track_ids())

    def test_jitter_single_frame_inside_does_not_alarm(self):
        s = IntrusionState(cfg())
        # tracker glitch: one frame the box jumps into the lane then back out
        xs = [300, 300, 300, 950, 300, 300, 300, 300]
        ys = [600, 620, 640, 660, 680, 700, 720, 740]
        ev = drive(s, 40, xs, ys)
        self.assertEqual([e for e in ev if e.kind == "alarm"], [])


if __name__ == "__main__":
    unittest.main()
