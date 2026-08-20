def build_extracted_text(
    detections
):

    return "\n".join(
        detection.get(
            "text",
            "",
        ).strip()
        for detection in detections
        if detection.get(
            "text",
            "",
        ).strip()
    )


def average_confidence(
    detections
):

    values = []

    for detection in detections:

        try:

            values.append(
                float(
                    detection.get(
                        "confidence",
                        0.0,
                    )
                )
            )

        except Exception:
            pass

    if not values:
        return 0.0

    return sum(values) / len(values)