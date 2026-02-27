"""
tools/region_picker.py – visually pick screen regions

Usage:
    python tools/region_picker.py

Instructions:
    1. Takes a screenshot of your primary monitor
    2. Shows it in a window
    3. Click and drag to draw a rectangle
    4. Press ENTER to confirm — prints (x, y, w, h) to console
    5. Draw another region or press ESC / Q to quit
"""

import cv2
import mss
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ── State ─────────────────────────────────────────────────────────────

state = {
    "drawing": False,
    "start_x": 0, "start_y": 0,
    "end_x": 0, "end_y": 0,
    "scale": 1.0,
}
regions: list[tuple[int, int, int, int]] = []  # stored in full-image coords
original: np.ndarray | None = None


def mouse_callback(event, x, y, flags, param):
    # With WINDOW_NORMAL, OpenCV already maps mouse coords to image space
    if event == cv2.EVENT_LBUTTONDOWN:
        state["drawing"] = True
        state["start_x"], state["start_y"] = x, y
        state["end_x"], state["end_y"] = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if state["drawing"]:
            state["end_x"], state["end_y"] = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        state["drawing"] = False
        state["end_x"], state["end_y"] = x, y


def main():
    global original

    # Capture screenshot
    with mss.mss() as sct:
        monitor = sct.monitors[config.MONITOR_INDEX]
        raw = sct.grab(monitor)
        screenshot = np.array(raw)
        original = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # Scale down if the image is too large for the display
    h, w = original.shape[:2]
    scale = 1.0
    max_win = 1280
    if w > max_win:
        scale = max_win / w
    display_w = int(w * scale)
    display_h = int(h * scale)
    state["scale"] = scale

    win_name = "Region Picker — drag to select, ENTER to confirm, ESC to quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, display_w, display_h)
    cv2.setMouseCallback(win_name, mouse_callback)

    print("\n=== Region Picker ===")
    print("  - Click and drag to draw a rectangle")
    print("  - Press ENTER to confirm the region")
    print("  - Press ESC or Q to quit\n")

    while True:
        display = original.copy()

        # Draw previously confirmed regions in green
        for (rx, ry, rw, rh) in regions:
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            label = f"({rx}, {ry}, {rw}, {rh})"
            cv2.putText(display, label, (rx, ry - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw current selection in blue
        sx, sy = state["start_x"], state["start_y"]
        ex, ey = state["end_x"], state["end_y"]
        if state["drawing"] or (sx != ex and sy != ey):
            cv2.rectangle(display, (sx, sy), (ex, ey), (255, 100, 0), 2)
            # Show live dimensions
            sel_x = min(sx, ex)
            sel_y = min(sy, ey)
            sel_w = abs(ex - sx)
            sel_h = abs(ey - sy)
            live_label = f"({sel_x}, {sel_y}, {sel_w}, {sel_h})"
            cv2.putText(display, live_label, (sel_x, sel_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(30) & 0xFF

        # ENTER — confirm current selection
        if key == 13:
            if sx != ex and sy != ey:
                # Normalise so x,y is always top-left (already in full-image coords)
                rx = min(sx, ex)
                ry = min(sy, ey)
                rw = abs(ex - sx)
                rh = abs(ey - sy)
                regions.append((rx, ry, rw, rh))
                print(f"  Region #{len(regions)}:  ({rx}, {ry}, {rw}, {rh})")
                # Reset selection
                state["start_x"] = state["start_y"] = 0
                state["end_x"] = state["end_y"] = 0

        # ESC or Q — quit
        elif key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()

    if regions:
        print("\n── Copy these into config.py ──")
        for i, (rx, ry, rw, rh) in enumerate(regions, 1):
            print(f"REGION_{i} = ({rx}, {ry}, {rw}, {rh})")
    print()


if __name__ == "__main__":
    main()
