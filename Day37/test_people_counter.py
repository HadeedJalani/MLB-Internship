
import sys, types
fake = types.ModuleType("ultralytics")
fake.YOLO = object
sys.modules["ultralytics"] = fake

from people_counter import CountingLine, CountingState, PersonTrack, ROI

shape = (100, 100, 3)
state = CountingState(
    line=CountingLine("horizontal", 0.5),
    roi=ROI(0, 1, 0, 1),
)

p = PersonTrack(7, 0.95, (40, 20, 60, 40))
state.update([p], shape)
assert state.current_count == 1
assert state.peak_count == 1
assert state.total_people_seen == 1
assert state.line_crossings == 0

p2 = PersonTrack(7, 0.95, (40, 60, 60, 80))
state.update([p2], shape)
assert state.line_crossings == 1
assert state.entries == 1

p3 = PersonTrack(7, 0.95, (40, 65, 60, 85))
state.update([p3], shape)
assert state.line_crossings == 1

print("Day37 state-machine smoke tests passed.")
