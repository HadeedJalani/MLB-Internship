from __future__ import annotations

import sys
from types import ModuleType

if 'ultralytics' not in sys.modules:
    fake_ultralytics = ModuleType('ultralytics')
    class FakeYOLO:
        pass
    fake_ultralytics.YOLO = FakeYOLO
    sys.modules['ultralytics'] = fake_ultralytics

from traffic_violation import (
    RESTRICTED_ZONE,
    WRONG_WAY,
    RegionBox,
    RoadDirection,
    ViolationState,
)
from tracker import MotionHistory, TrackedVehicle


def vehicle(track_id: int, x: int, y: int, name: str = "car") -> TrackedVehicle:
    return TrackedVehicle(
        track_id=track_id,
        class_id=2,
        class_name=name,
        confidence=0.9,
        box=(x - 10, y - 10, x + 10, y + 10),
    )


def test_zone_is_edge_triggered() -> None:
    state = ViolationState(
        RoadDirection("down"),
        RegionBox(0.40, 0.60, 0.40, 0.60),
    )
    motion = MotionHistory(length=4)

    # First observation inside initializes state; no false entry event.
    v = vehicle(1, 50, 50)
    motion.update(v.track_id, v.centroid)
    state.update([v], motion, (100, 100, 3), 0, 25.0)
    assert state.restricted_zone_count == 0

    # Still inside: no duplicate.
    for frame in range(1, 5):
        motion.update(v.track_id, v.centroid)
        state.update([v], motion, (100, 100, 3), frame, 25.0)
    assert state.restricted_zone_count == 0

    # Leave and re-enter: exactly one new zone event.
    out = vehicle(1, 10, 10)
    motion.update(out.track_id, out.centroid)
    state.update([out], motion, (100, 100, 3), 5, 25.0)

    back = vehicle(1, 50, 50)
    motion.update(back.track_id, back.centroid)
    state.update([back], motion, (100, 100, 3), 6, 25.0)
    assert state.restricted_zone_count == 1
    assert state.events[-1].violation_type == RESTRICTED_ZONE


def test_wrong_way_is_confirmed_once() -> None:
    direction = RoadDirection(
        "down",
        angle_threshold_deg=120.0,
        confirm_window=4,
        confirm_ratio=0.75,
    )
    state = ViolationState(direction)
    motion = MotionHistory(length=4)

    # Move upward while normal traffic is down.
    for frame, y in enumerate([80, 65, 50, 35]):
        v = vehicle(7, 50, y)
        motion.update(v.track_id, v.centroid)
        state.update([v], motion, (100, 100, 3), frame, 25.0)

    assert state.wrong_way_count == 1
    assert 7 in state.wrong_way_ids

    # More wrong-way frames cannot create a second wrong-way event.
    for frame, y in enumerate([25, 15, 5], start=4):
        v = vehicle(7, 50, y)
        motion.update(v.track_id, v.centroid)
        state.update([v], motion, (100, 100, 3), frame, 25.0)

    assert state.wrong_way_count == 1
    assert sum(e.violation_type == WRONG_WAY for e in state.events) == 1


if __name__ == "__main__":
    test_zone_is_edge_triggered()
    test_wrong_way_is_confirmed_once()
    print("All Day36 rule-engine tests passed.")
