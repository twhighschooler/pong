import time
import os
import keyboard
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
lr_actor = 1e-5#changed
lr_critic = 1e-4#changed
gamma = 0.99

tau = 0.001
noise_decay = 0.999
min_noise = 0.05
exploration_noise = 0.1
batch_size = 64
buffer_size = 10000
stop_training = False
emergency_stop_key = 'esc'
frame_duration = 1 / fps
monitor = {"top": 227, "left": 560, "width": 800, "height": 600}
frame_stack = deque(maxlen=4)  # Stack to hold the last 4 frames

sct = mss.mss()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_emergency_stop():
    """Set up emergency stop key listener"""
    global stop_training

    def on_stop_key():
        global stop_training
        stop_training = True
        print("interrupt by keyboard,you pressed esc,dumbass")
    keyboard.add_hotkey(emergency_stop_key, on_stop_key)

def check_stop_condition():
    return stop_training
def get_game_status():
    try:

        finished = False
        playing = False
        with open(r'C:\Program Files (x86)\StreamCompanion\Files\finished.txt')as f:
            status = f.read().strip()
            if status =="ResultsScreen":
                finished = True
                playing = False
            elif status == "Playing":
                playing = True
                finished = False

        with open(r'C:\Program Files (x86)\StreamCompanion\Files\300.txt') as f: 
            try:
                perfects = int(f.read().strip())
            except Exception:
                perfects = 0
        with open(r'C:\Program Files (x86)\StreamCompanion\Files\100.txt') as f:
            try:
                goods = int(f.read().strip())
            except Exception:
                goods = 0
        with open(r'C:\Program Files (x86)\StreamCompanion\Files\50.txt') as f:
            try:
                bads = int(f.read().strip())
            except Exception:
                bads = 0
        with open(r'C:\Program Files (x86)\StreamCompanion\Files\misses.txt') as f:
            try:
                misses = int(f.read().strip())
            except Exception:
                misses = 0
        with open(r'C:\Program Files (x86)\StreamCompanion\Files\combo.txt') as f:
            try:
                combo = int(f.read().strip())
            except Exception:
                combo = 0
        return {
            'finished':finished,
            'playing':playing,
            'perfects': perfects,
            'goods':goods,
            'bads':bads,
            'misses':misses,
            'combo':combo
        }
    except Exception as e:
        print(f"Error reading :{e}")
        return{
            'finished':False,
            'playing':False,
            'perfects': 0,
            'goods':0,
            'bads':0,
            'misses':0,
            'combo':0
        }

def calculate_reward(prev_stats, current_stats):

    reward = 0

    new_perfects = current_stats['perfects'] - prev_stats['perfects']
    new_goods = current_stats['goods'] - prev_stats['goods']    
    new_bads = current_stats['bads'] - prev_stats['bads']
    new_misses = current_stats['misses'] - prev_stats['misses']

    if new_perfects > 5 or new_perfects < 0:
        new_perfects = 0
    if new_goods > 5 or new_goods < 0:
        new_goods = 0
    if new_bads > 5 or new_bads < 0:
        new_bads = 0
    if new_misses > 5 or new_misses < 0:
        new_misses = 0

    reward += new_perfects * 2
    reward += new_goods * 2
    reward -= new_bads * 0.4
    reward -= new_misses * 4
    
    if current_stats['combo'] >50:
        reward += 5
    
    

    return reward

def initialize_frame_stack():
    initialize_frame =capture_screen()
    frame_stack.clear
    for _ in range(4):
        frame_stack.append(initialize_frame)
    return update_stack(initialize_frame)
    

def mouse_position():
    x, y = pyautogui.position()
    mouse_x_norm = max(-1,min(1,(x - monitor["left"]) / monitor["width"]*2 - 1))
    mouse_y_norm = max(-1, min(1, (y - monitor["top"]) / monitor["height"] * 2 - 1))
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
        stacked = np.stack(padded_stack, axis = 0)
    else:
        stacked = np.stack(frame_stack, axis=0)
    tensor = torch.tensor(stacked, dtype=torch.float32,device=device) 
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    return tensor
def capture_screen():
    
    start = time.perf_counter()
    img = sct.grab(monitor)
    processed =  preprocess_frame(img)
    
    return processed
    
class ReplayBuffer:
    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    def push(self,frames,mouse_pos , action, reward, next_frames,next_mouse_pos, done):
        self.buffer.append((frames,mouse_pos, action, reward, next_frames,next_mouse_pos, done))
    def sample(self,batch_size):
        batch = random.sample(self.buffer, batch_size)
        frames,mouse_pos, action, reward, next_frames,next_mouse_pos, done = zip(*batch)

        frames = torch.cat(frames,dim = 0)
        mouse_pos = torch.cat(mouse_pos,dim = 0)
        action = torch.cat(action, dim=0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(device)
        next_frames = torch.cat(next_frames,dim=0)
        next_mouse_pos = torch.cat(next_mouse_pos,dim=0)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(device)
        return frames,mouse_pos, action, reward, next_frames,next_mouse_pos, done
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
        
        self.mouse_fc = nn.Sequential(
            nn.Linear(2,256),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 9 + 256, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, action_space_size),
            nn.Tanh()  # Output action in range [-1, 1]
        )

        nn.init.xavier_uniform_(self.fc[-2].weight, gain=0.1)
        nn.init.constant_(self.fc[-2].bias, 0)
    def forward(self,frame, mouse_pos):
        x= self.conv(frame)
        x = torch.flatten(x,1)
        mouse_features= self.mouse_fc(mouse_pos)
        x = torch.cat([x,mouse_features], dim=1)
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
        
        self.mouse_fc = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU()
            )
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 9 + 256+ action_space_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1)  # Q-value
        )
    def forward(self,img, mouse_pos,action):
        x= self.conv(img)
        x = torch.flatten(x,1)
        mouse_features = self.mouse_fc(mouse_pos)
        x = torch.cat([x, mouse_features, action], dim=1)
        
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
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic,weight_decay=1e-5)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        
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
            noise = torch.randn_like(action) * self.exploration_noise
            action = action + noise 
            action = torch.clamp(action, -1, 1)

        return action
    def update(self,current_stats):
        # print('update being used')
        if len(self.replay_buffer) < batch_size:
            return 
        frames,mouse_pos,action,reward,next_frames,next_mouse_pos,done=self.replay_buffer.sample(batch_size)
        
        if not current_stats['playing']:
            return 
        with torch.no_grad():
            next_actions = self.target_actor(next_frames, next_mouse_pos)
            next_q = self.target_critic(next_frames, next_mouse_pos, next_actions)
            target_q = reward + (gamma * next_q * (1 - done))
        
        current_q = self.critic(frames, mouse_pos,action)
        # print(f"current_q: min={current_q.min().item():.2f}, max={current_q.max().item():.2f}, mean={current_q.mean().item():.2f}")
        # print(f"target_q: min={target_q.min().item():.2f}, max={target_q.max().item():.2f}, mean={target_q.mean().item():.2f}")

        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),max_norm=1.0)
        self.critic_optimizer.step()

        predicted_actions = self.actor(frames, mouse_pos)
        actor_loss = -self.critic(frames, mouse_pos, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) 
        self.actor_optimizer.step()

        self.soft_update(self.target_actor, self.actor, tau)
        self.soft_update(self.target_critic, self.critic, tau)

        self.exploration_noise = max(min_noise, self.exploration_noise * noise_decay)

        return critic_loss.item(), actor_loss.item()

def training_loop():
    ddpg = DDPG(action_space_size=2)
    global next_frame_time,frame_stack
    setup_emergency_stop()
    
    episode = 0
    


    print("Starting training ...")
    try:
        while not check_stop_condition():
            episode += 1
            
            random_x = monitor["left"] + random.randint(100,monitor["width"] - 100)
            random_y = monitor["top"] + random.randint(100, monitor["height"] - 100)



            step = 0
            print(f"Starting Episode {episode}")

            while not check_stop_condition():
                current_stats = get_game_status()
                if current_stats['playing']:  # Game is active
                    break
                time.sleep(0.1)
            print('game started!')
            state_frames = initialize_frame_stack()
            prev_stats = get_game_status()
            
            prev_mouse_pos = mouse_position()

            while not check_stop_condition():
                step += 1
                
                

                action = ddpg.select_action(state_frames, prev_mouse_pos)

                action_np = action.cpu().numpy().flatten()
                penalty_threshold = 0.95
                penalty=0
                if any(abs(a) >= penalty_threshold for a in action_np):
                    penalty = -5
                # print("Action output:", action_np)

                mouse_x_norm = action_np[0]
                mouse_y_norm = action_np[1]
                
                new_x = monitor["left"] + (mouse_x_norm + 1) / 2 * monitor["width"]
                new_y = monitor["top"]  + (mouse_y_norm + 1) / 2 * monitor["height"]
                
                new_x = max(monitor["left"], min(monitor["left"] + monitor["width"] - 1, new_x))
                new_y = max(monitor["top"], min(monitor["top"] + monitor["height"] - 1, new_y))
                
                pyautogui.moveTo(new_x, new_y, duration=0.04)
            
                # below is the precise fps control  
                next_frame_time += frame_duration
                sleep_time = next_frame_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # we're late, skip sleeping to catch up
                    next_frame_time = time.perf_counter()

                next_frame = capture_screen()
                next_state_frames = update_stack(next_frame)
                current_mouse_pos = mouse_position()
                current_stats = get_game_status()
                done = current_stats['finished']
                if done:
                    print(f"Episode {episode} finished!")
                    break
                reward = calculate_reward(prev_stats, current_stats)
                reward = reward + penalty

                

                ddpg.replay_buffer.push(state_frames,prev_mouse_pos,action,reward,next_state_frames,current_mouse_pos,done)
               
                
                if len(ddpg.replay_buffer) >= batch_size and step % 4 ==0:
                    losses = ddpg.update(current_stats)
                    if losses and step % 100 == 0:
                        critic_loss, actor_loss = losses
                        print (f"step:{step}, Critic Loss:{critic_loss},Actor_loss:{actor_loss}")
                        
            
                state_frames = next_state_frames
                prev_mouse_pos = current_mouse_pos
                prev_stats = current_stats.copy()

                
            
            if check_stop_condition():
                break

            print(f"Episode: {episode},perfects:{current_stats['perfects']},goods:{current_stats['goods']},bads:{current_stats['bads']},misses:{current_stats['misses']} ,Noise: {ddpg.exploration_noise:.4f}")
            time.sleep(1)
            pyautogui.moveTo(634,799)
            pyautogui.click() 

            time.sleep(1)
            pyautogui.moveTo(879,795)
            pyautogui.click()

            time.sleep(5)
            pyautogui.moveTo(1301,792)
            pyautogui.click()

            if episode % 50 == 0:
                torch.save({ 
                    'episode': episode,
                    'exploration_noise' : ddpg.exploration_noise,
                    'actor_state_dict': ddpg.actor.state_dict(),
                    'critic_state_dict': ddpg.critic.state_dict(),
                    'actor_optimizer': ddpg.actor_optimizer.state_dict(),
                    'critic_optimizer': ddpg.critic_optimizer.state_dict(),
                }, f'ddpg_checkpoint_episode_{episode}.pth')
                print(f"Saved checkpoint at episode {episode}")
    except KeyboardInterrupt:
        print("\n keyboard interrupt by you")
        stop_training = True
    finally:
        print('stopped and saved(probably)')
        if episode >5:
            torch.save({ 
                    'episode': episode,
                    'exploration_noise' : ddpg.exploration_noise,
                    'actor_state_dict': ddpg.actor.state_dict(),
                    'critic_state_dict': ddpg.critic.state_dict(),
                    'actor_optimizer': ddpg.actor_optimizer.state_dict(),
                    'critic_optimizer': ddpg.critic_optimizer.state_dict(),
                }, f'ddpg_checkpoint_episode_{episode}.pth')
        stop_training = False

if __name__ == "__main__":
    # Uncomment to start training
    training_loop()
    print("DDPG implementation ready. Uncomment training_loop() to start training.")
    print("Make sure osu! and StreamCompanion are running before starting training.")