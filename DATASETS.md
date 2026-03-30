# Datasets

This project supports:
- EPIC-KITCHENS
- Ego4D
- DROID
- BridgeData V2
- Something-Something V2
- toy synthetic dataset

All loaders sample clips of length `T` and return:
- `frames`: `[B, T, C, H, W]`
- `actions`: `[B, T-1, A]`

If a dataset path is missing or malformed, loaders fail with explicit messages that include expected layout.

## Expected Layouts

### EPIC-KITCHENS

```text
EPIC-KITCHENS/
  P01/
    rgb_frames/
      P01_01/
        frame_0000000001.jpg
```

### Ego4D

```text
ego4d/
  videos/
    *.mp4
```

### DROID

```text
<tfds_root>/droid/<version>/
```

Requirements:
- `tensorflow`
- `tensorflow-datasets`

### BridgeData V2

```text
bridge_data_v2/
  trajectories/
    <episode_id>/
      images/*.jpg
      actions.npy (optional)
```

### Something-Something V2

```text
something-something-v2/
  videos/
    *.webm or *.mp4
```

## Phase Notes

- Phase 1 includes functional loaders and clear failures.
- TODO(phase2): enrich dataset-specific parsing (official metadata, canonical splits, stronger action semantics).
