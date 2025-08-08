import time
import collections
import numpy as np
import mss
import cv2

next_frame_time = time.perf_counter()
fps = 25
frame_duration = 1 / fps
monitor = {"top": 227, "left": 560, "width": 800, "height": 600}

sct = mss.mss()
while True:
    start = time.perf_counter()
    img = sct.grab(monitor)
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    

    next_frame_time += frame_duration
    sleep_time = next_frame_time - time.perf_counter()
    if sleep_time > 0:
        time.sleep(sleep_time)
    else:
        # we're late, skip sleeping to catch up
        next_frame_time = time.perf_counter()
