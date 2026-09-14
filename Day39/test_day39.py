from security_monitoring import (
    normalize_events,
    point_inside_roi,
)


def test_point_inside_roi() -> None:
    roi = (0, 0, 100, 100)

    assert point_inside_roi(50, 50, roi)
    assert point_inside_roi(0, 0, roi)
    assert point_inside_roi(100, 100, roi)

    assert not point_inside_roi(101, 50, roi)
    assert not point_inside_roi(50, 101, roi)


def test_point_without_roi() -> None:
    assert point_inside_roi(500, 500, None)


def test_event_dataframe_dtypes() -> None:
    events = [
        [
            7,
            "ENTRY",
            "ROI",
            10,
            0.40,
            0.40,
            None,
            None,
            "active",
        ],
        [
            7,
            "EXIT",
            "ROI",
            30,
            1.20,
            0.40,
            1.20,
            0.80,
            "completed",
        ],
    ]

    dataframe = normalize_events(events)

    assert str(dataframe["track_id"].dtype) == "int64"
    assert str(dataframe["frame"].dtype) == "int64"
    assert str(dataframe["timestamp_s"].dtype) == "float64"
    assert str(dataframe["exit_time_s"].dtype) == "float64"
    assert str(dataframe["status"].dtype) == "string"


def test_empty_event_dataframe() -> None:
    dataframe = normalize_events([])

    assert dataframe.empty
    assert "track_id" in dataframe.columns
    assert "event" in dataframe.columns
    assert "duration_s" in dataframe.columns


if __name__ == "__main__":
    test_point_inside_roi()
    test_point_without_roi()
    test_event_dataframe_dtypes()
    test_empty_event_dataframe()

    print("Day 39 tests passed successfully.")