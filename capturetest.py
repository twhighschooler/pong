import time
import torchvision
from torch import nn
import torch
import numpy as np
import mss
import cv2
from collections import deque
import pyautogui

next_frame_time = time.perf_counter()
fps = 25
frame_duration = 1 / fps
monitor = {"top": 227, "left": 560, "width": 800, "height": 600}
frame_stack = deque(maxlen=4)  # Stack to hold the last 4 frames
frame_stack.clear()  # Clear the stack initially
sct = mss.mss()

def mouse_position():
    x, y = pyautogui.position()
    mouse_pos_np =  np.array([x, y], dtype=np.float32)
    mouse_pos_tensor = torch.from_numpy(mouse_pos_np).float()
    mouse_pos_tensor = mouse_pos_tensor.unsqueeze(0)  # Add batch dimension
    return mouse_pos_tensor


def preprocess_frame(img):
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    frame = cv2.resize(frame, dsize=(80, 60))
    frame = frame / 255.0# Normalize to [0, 1]
    return frame
def update_stack(new_frame):
    frame_stack.append(new_frame)
    stacked = np.stack(frame_stack, axis=0)
    tensor = torch.tensor(stacked, dtype=torch.float32) / 255.0
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    tensor = tensor.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return tensor
def capture_screen():
    
    start = time.perf_counter()
    img = sct.grab(monitor)
    processed =  preprocess_frame(img)
    
    return processed
    # below is the precise fps control  
    # next_frame_time += frame_duration
    # sleep_time = next_frame_time - time.perf_counter()
    # if sleep_time > 0:
    #     time.sleep(sleep_time)
    # else:
    #     # we're late, skip sleeping to catch up
    #     next_frame_time = time.perf_counter()
class Actor(nn.Module):
    def __init__(self, action_space_size=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, 5, 3),
            nn.ReLU(),
            nn.Conv2d(64,128, 5, 2),
            nn.ReLU(),
            nn.Conv2d(128,128, 3, 1),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 9 + 2, 2048),
            nn.ReLU(),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
        )
    def forward(self,frame, mouse_pos):
        x= self.conv(frame)
        x = torch.flatten(x,1)
        x = self.fc_normal(x)
        x = torch.cat([x, mouse_pos], dim=1)
        return self.fc(x)
    
class Critic(nn.Module):
    def __init__(self, action_space_size=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, 5, 3),
            nn.ReLU(),
            nn.Conv2d(64,128, 5, 2),
            nn.ReLU(),
            nn.Conv2d(128,128, 3, 1),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 9 + 2, 2048),
            nn.ReLU(),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024, 1)  # Q-value
        )
    def forward(self,img, mouse_pos,action):
        x= self.conv(img)
        x = torch.flatten(x,1)
        x = self.fc_normal(x)
        x = torch.cat([x, mouse_pos,action], dim=1)
        return self.fc(x)
                



    