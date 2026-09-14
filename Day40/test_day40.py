from video_analytics import (
    VideoConfig,
    build_roi,
    normalize_event_dataframe,
    point_inside_roi,
)


def test_roi_creation():
    config = VideoConfig(
        roi_x1=0.10,
        roi_y1=0.20,
        roi_x2=0.80,
        roi_y2=0.90,
    )

    assert build_roi(1000, 500, config) == (
        100,
        100,
        800,
        450,
    )


def test_roi_membership():
    roi = (100, 100, 800, 450)

    assert point_inside_roi(200, 200, roi)
    assert not point_inside_roi(900, 200, roi)


def test_event_dataframe():
    events = [
        [1, "ENTRY", 100, 4.0, 4.0, None, None, "active"],
        [1, "EXIT", 200, 8.0, 4.0, 8.0, 4.0, "completed"],
    ]

    df = normalize_event_dataframe(events)

    assert len(df) == 2
    assert df["track_id"].dtype.name == "int64"
    assert df["timestamp_s"].dtype.name == "float64"
    assert df["event"].dtype.name == "string"


def test_empty_dataframe():
    df = normalize_event_dataframe([])
    assert df.empty
    assert "track_id" in df.columns
    assert "event" in df.columns


if __name__ == "__main__":
    test_roi_creation()
    test_roi_membership()
    test_event_dataframe()
    test_empty_dataframe()
    print("Day 40 tests passed successfully.")
