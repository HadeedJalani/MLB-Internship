# Sample traffic videos

Add 3–5 public traffic videos here for the internship demo.

Recommended sources:

- Pexels Videos: https://www.pexels.com/videos/
- Pixabay Videos: https://pixabay.com/videos/
- Public traffic / MOT datasets:
  - UA-DETRAC: https://detrac-db.rit.albany.edu/
  - BDD100K: https://www.vis.xyz/bdd100k/

Choose clips with:

- visible moving vehicles
- reasonably stable camera
- enough frames for tracking
- different camera/road directions
- limited blur and extreme darkness

## Calibration files

For each sample, you can add:

```text
my_traffic.mp4
my_traffic_config.json
```

Example:

```json
{
  "direction": {
    "name": "down",
    "angle_threshold_deg": 125,
    "confirm_window": 12,
    "confirm_ratio": 0.75
  },
  "restricted_zone": {
    "x_min": 0.70,
    "x_max": 0.95,
    "y_min": 0.40,
    "y_max": 0.95
  },
  "direction_roi": {
    "x_min": 0.05,
    "x_max": 0.95,
    "y_min": 0.10,
    "y_max": 0.95
  }
}
```

Coordinates are fractions of frame width/height.

Do not commit a video unless its license permits redistribution in your GitHub repository.
