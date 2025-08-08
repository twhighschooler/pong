import time
import torchvision
from torch import nn
import torch
import numpy as np
import mss
import cv2
from collections import deque
import torch.optim as optim
import pyautogui
import random
import torch.nn.functional as F

next_frame_time = time.perf_counter()
fps = 25
lr_actor = 1e-4
lr_critic = 1e-3
gamma = 0.99
tau = 0.001
noise_decay = 0.995
exploration_noise = 0.1
batch_size = 64
buffer_size = 10000
frame_duration = 1 / fps
monitor = {"top": 227, "left": 560, "width": 800, "height": 600}
frame_stack = deque(maxlen=4)  # Stack to hold the last 4 frames
frame_stack.clear()  # Clear the stack initially
sct = mss.mss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mouse_position():
    x, y = pyautogui.position()
    mouse_x_norm = (x - monitor["left"]) / monitor["width"]
    mouse_y_norm = (y - monitor["top"]) / monitor["height"]
    mouse_pos_np =  np.array([mouse_x_norm, mouse_y_norm], dtype=np.float32)
    mouse_pos_tensor = torch.from_numpy(mouse_pos_np).float()
    mouse_pos_tensor = mouse_pos_tensor.unsqueeze(0)  # Add batch dimension
    return mouse_pos_tensor.to(device)


def preprocess_frame(img):
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    frame = cv2.resize(frame, dsize=(80, 60))
    frame = frame / 255.0# Normalize to [0, 1]
    return frame
def update_stack(new_frame):
    frame_stack.append(new_frame)
    if len(frame_stack) < 4:
        padded_stack = [np.zeros((60,80))] * (4- len(frame_stack))+ list(frame_stack)
        frame_stack.append(np.zeros_like(new_frame))
    else:
        stacked = np.stack(frame_stack, axis=0)
    tensor = torch.tensor(stacked, dtype=torch.float32) 
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    tensor = tensor.to(device)
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

class ReplayBuffer:
    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    def push(self,state , action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self,batch_size):
        batch = random.samle(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.cat(state, dim=0)
        action = torch.cat(action, dim=0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(device)
        next_state = torch.cat(next_state, dim=0)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(device)
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)
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
            nn.Linear(1024, action_space_size),
        )
    def forward(self,frame, mouse_pos):
        x= self.conv(frame)
        x = torch.flatten(x,1)
        
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
        
        x = torch.cat([x, mouse_pos,action], dim=1)
        return self.fc(x)
    
class DDPG:
    def __init__(self, action_space_size=2):
        self.action_space_size = action_space_size
        self.actor = Actor(action_space_size).to(device)
        self.critic = Critic(action_space_size).to(device)
        self.target_actor = Actor(action_space_size).to(device)
        self.target_critic = Critic(action_space_size).to(device)            
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.noise = np.random.normal(0,0.1, size=(action_space_size))
        self.exploration_noise =    exploration_noise
    def hard_update(self,target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    def soft_update(self,target, source, tau=tau):
        for target_param,param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    def select_action(self, state, mouse_pos,add_noise=True):
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state, mouse_pos)
        self.actor.train()
        if add_noise:
            noise = torch.tensor(self.noise.sample(), dtype=torch.float32).to(device)
            action = action + noise * self.exploration_noise
            action = torch.clamp(action, -1, 1)

        return action
    def update(self):
        if len(self.replay_buffer < batch_size):
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(batch_size)
        state_frames = state_batch[:, :4, :, :]  # First 4 channels are frames
        state_mouse = state_batch[:, 4:6, 0, 0]  # Mouse pos stored in channels 4-5
        next_state_frames = next_state_batch[:, :4, :, :]
        next_state_mouse = next_state_batch[:, 4:6, 0, 0]

        with torch.no_grad():
            next_actions = self.target_actor(next_state_frames, next_state_mouse)
            next_q = self.target_critic(next_state_frames, next_state_mouse, next_actions)
            target_q = reward_batch + (gamma * next_q * (1 - done_batch))
        
        current_q = self.critic(state_frames, state_mouse, action_batch)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        predicted_actions = self.actor(state_frames, state_mouse)
        actor_loss = -self.critic(state_frames, state_mouse, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.target_actor, self.actor, tau)
        self.soft_update(self.target_critic, self.critic, tau)
    