"""Intrusion state machine.

Consumes tracked vehicle positions per frame and decides when to alarm.
Pure Python, no CV dependencies, fully unit testable.

Logic (design doc section 10):
  1. Anchor point is the bottom centre of the bounding box.
  2. Containment: anchor inside the closed lane polygon.
  3. Heading: displacement over the last N frames has a component toward the crew.
  4. Confirmation: alarm only after `confirm_frames` consecutive inside+inbound frames.
  5. Exit grace: a track that leaves before confirmation is a near miss, not an alarm.
  6. Cooldown: one alarm per track, and a global cooldown between alarms.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .geometry import Point, dot, normalize, point_in_polygon


@dataclass
class LaneConfig:
    polygon: list[Point]              # image coordinates, in order
    crew_direction: Point             # unit vector in image space pointing toward the crew
    confirm_frames: int = 4
    heading_window: int = 5
    heading_min_speed_px: float = 2.0  # px per frame along crew_direction to count as inbound
    cooldown_s: float = 8.0
    track_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        self.crew_direction = normalize(self.crew_direction)
        if len(self.polygon) < 3:
            raise ValueError("polygon needs at least 3 points")


@dataclass
class Event:
    kind: str                 # "alarm" or "near_miss"
    track_id: int
    t: float
    anchor: Point
    frames_inside: int
    speed_px_per_frame: float


@dataclass
class _Track:
    history: deque = field(default_factory=lambda: deque(maxlen=16))  # (t, anchor)
    inside_run: int = 0
    was_inside: bool = False
    alarmed: bool = False
    last_seen: float = 0.0


class IntrusionState:
    def __init__(self, cfg: LaneConfig):
        self.cfg = cfg
        self._tracks: dict[int, _Track] = {}
        self._pending: list[Event] = []
        self._last_alarm_t: float | None = None

    # ---- public API -------------------------------------------------------

    def update(self, track_id: int, bbox_xyxy: tuple[float, float, float, float], t: float) -> None:
        """Feed one tracked detection for the current frame at time t (seconds)."""
        x1, y1, x2, y2 = bbox_xyxy
        anchor = ((x1 + x2) / 2.0, y2)
        tr = self._tracks.setdefault(track_id, _Track())
        tr.history.append((t, anchor))
        tr.last_seen = t

        inside = point_in_polygon(anchor, self.cfg.polygon)
        speed_along = self._speed_toward_crew(tr)
        inbound = speed_along >= self.cfg.heading_min_speed_px

        if inside and inbound:
            tr.inside_run += 1
        else:
            if tr.was_inside and not inside and 0 < tr.inside_run < self.cfg.confirm_frames and not tr.alarmed:
                self._pending.append(Event("near_miss", track_id, t, anchor, tr.inside_run, speed_along))
            tr.inside_run = 0

        tr.was_inside = inside

        if tr.inside_run >= self.cfg.confirm_frames and not tr.alarmed:
            if self._cooldown_ok(t):
                tr.alarmed = True
                self._last_alarm_t = t
                self._pending.append(Event("alarm", track_id, t, anchor, tr.inside_run, speed_along))

    def end_frame(self, t: float) -> None:
        """Call once per frame after all updates to expire stale tracks."""
        stale = [tid for tid, tr in self._tracks.items() if t - tr.last_seen > self.cfg.track_timeout_s]
        for tid in stale:
            del self._tracks[tid]

    def pop_events(self) -> list[Event]:
        out, self._pending = self._pending, []
        return out

    def track_ids(self) -> list[int]:
        return list(self._tracks)

    # ---- internals --------------------------------------------------------

    def _speed_toward_crew(self, tr: _Track) -> float:
        h = tr.history
        if len(h) < 2:
            return 0.0
        window = min(self.cfg.heading_window, len(h) - 1)
        (t0, p0), (t1, p1) = h[-1 - window], h[-1]
        disp = (p1[0] - p0[0], p1[1] - p0[1])
        return dot(disp, self.cfg.crew_direction) / window

    def _cooldown_ok(self, t: float) -> bool:
        return self._last_alarm_t is None or (t - self._last_alarm_t) >= self.cfg.cooldown_s
