# Duck Hunt Gesture Controller 🔫

Control a Duck Hunt game using hand gestures detected via your webcam!

## Features

- **Gesture-based aim**: Point with **index finger** to aim (barrel)
- **Thumb trigger**: Fold thumb to palm to shoot, extend outward to ready
- **Smooth tracking**: 1€ filter + 3D distance calculation for stable detection
- **Calibration system**: Personalize thumb-to-palm thresholds for your hand

## Quick Start (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run calibration
```bash
python calibration.py
```
- Press **E** 5 times with thumb **EXTENDED** (Ready position)
- Press **F** 5 times with thumb **FOLDED** to palm (Fire position)
- Game starts automatically after calibration

### 3. Or run controller directly (if already calibrated)
```bash
python duck_hunt_controller.py
```

## Controls

| Key | Action |
|-----|--------|
| Q | Quit |
| D | Toggle display window |
| R | Reset filters |
| T | Toggle "Always on Top" |

## Gestures

```
    👍 Thumb EXTENDED outward = Ready (don't shoot)
    👊 Thumb TOUCHING palm  = Fire!
    👆 Index finger         = Aim (cursor)
```

## Files

| File | Description |
|------|-------------|
| `duck_hunt_controller.py` | Local game controller |
| `calibration.py` | Calibration tool |
| `utils.py` | Core detection classes |
| `calibration.txt` | Saved calibration data |

## Technical Details

### Optimizations
- **3D angle calculation**: Uses X, Y, Z coordinates to avoid wild readings when hand tilts
- **Temporal smoothing**: Median filter + 1€ filter for stable angles
- **Outlier rejection**: Ignores readings >3 std deviations from median
- **Confirmation-based trigger**: Requires 2+ consistent frames before firing/releasing
- **Velocity-based cursor easing**: Faster response for large movements

### Thresholds
- Fire threshold: 55% of calibrated range
- Release threshold: 35% of calibrated range
- Shot cooldown: 350ms
