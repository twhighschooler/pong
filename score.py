import os
import time


def get_game_status():
    try:
        with open('C:\Program Files (x86)\StreamCompanion\Files\finished.txt')as f:
            status = f.read()
            if status =="ResultsScreen":
                finished = True
            else:
                finished = False

        with open('C:\Program Files (x86)\StreamCompanion\Files\300.txt') as f: 
            try:
                perfects = int(f.read())
            except Exception:
                perfects = 0
        with open('C:\Program Files (x86)\StreamCompanion\Files\100.txt') as f:
            try:
                goods = int(f.read())
            except Exception:
                goods = 0
        with open('C:\Program Files (x86)\StreamCompanion\Files\50.txt') as f:
            try:
                bads = int(f.read())
            except Exception:
                bads = 0
        with open('C:\Program Files (x86)\StreamCompanion\Files\misses.txt') as f:
            try:
                misses = int(f.read())
            except Exception:
                misses = 0
        with open('C:\Program Files (x86)\StreamCompanion\Files\combo.txt') as f:
            try:
                combo = int(f.read())
            except Exception:
                combo = 0
    except Exception:
        print(Exception)

def calculate_reward(prev_stats, current_stats):

    reward = 0

    new_perfects = current_stats['perfects'] - prev_stats['perfects']
    new_goods = current_stats['goods'] - prev_stats['goods']    
    new_bads = current_stats['bads'] - prev_stats['bads']
    new_misses = current_stats['misses'] - prev_stats['misses']

    reward += new_perfects * 10
    reward += new_goods * 5  
    reward -= new_bads * 2
    reward -= new_misses * 10
    
    if current_stats['combo'] >50:
        reward += 5 
    
    if current_stats['finished']:
        reward += 100

    return reward

