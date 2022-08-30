# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 00:21:39 2022

@author: X2029440
"""

import torch 
import numpy as np 
import random 
from qnetwork import QNet, QTrainer
from collections import deque
import torch
import itertools

#class that defines an AI player
class Agent : 
  
  BATCH_SIZE = 100_000
  MAX_MEMOMERY = 200
  
  
  def __init__(self, bird) : 
    
    self.bird = bird  
    self.qnetwork = QNet(input_size=3, output_size=2) #notre q model 
    self.qtrainer = QTrainer(self.qnetwork, 1e-3, 0.9) #model, lr, gamma
    self.epsilon = 0 #tradeoff between exploration / exploitation 
    self.memory = deque(maxlen=Agent.MAX_MEMOMERY)
    
  

      
  #what is the action to take given a specific game_state for a game ? 
  #explore : whether or not to explore in the beginning (we may not want that if we use a loaded qnetwork for instance)
  #returns : action, vector of size 3 : ['left', 'straight','right'], ex : [1,0,0] means turning to your left 
  def get_action(self, game_state, nb_games, explore = True) : 

    # self.epsilon = 80 - nb_games if explore else 0 
    # action = [0,0]

    # if random.randint(0, 200) < self.epsilon and explore: #exploration (mouvement aléatoire)
    #   move = random.randint(0, 1)
    #   action[move] = 1
    
    
    action = [0,0]
    
    start_epsilon = 1 #(start proba of random)
    end_epsilon = 1 / 100 #(end proba of random)
    nb_games_end_epsilon = 1000 #nb of games to reach end epsilon 
    if nb_games > nb_games_end_epsilon : nb_games = nb_games_end_epsilon
    epsilon = start_epsilon - nb_games * (start_epsilon - end_epsilon) / nb_games_end_epsilon
    if random.uniform(0,1) < epsilon and explore: #exploration (mouvement aléatoire)
      print('exploration')
      ratio = 0.1 #prob of jump if exploration 
      if random.uniform(0,1) < ratio : move = 0 #jump
      else : move = 1 #do not jump 
      action[move] = 1
    

    else: #exploitation (we use the qnetwork model) 
      game_state_tensor = torch.tensor(game_state, dtype=torch.float)
      prediction = self.qnetwork(game_state_tensor)
      move = torch.argmax(prediction).item()
      action[move] = 1
    
    return action
    [170.0, -210.0, 688.0, 320.0]

  # x_distance next pipe (top left corner) (max=375, min=-75) max=375 au début puis 270 
  # y_distance next pipe (top left corner) (max=500, min=500)
  # bird_y_pos (to get info abt floor distance) (max=700, min=0) (i removed it)
  # bird_y_velocity (max=380, min=-150)
  def get_game_state(self, game) : 
    
    game_state = [0,0,0]
    
    if game.next_pipe != None :
      dst_pipe_x = game.next_pipe.x - self.bird.x
      dst_pipe_y = game.next_pipe.y - self.bird.y
    
      game_state[0] = dst_pipe_x
      game_state[1] = dst_pipe_y
      
    game_state[2] = self.bird.y_vel
    
    #we min max scale each value 
    # game_state[0] = dst_pipe_x #round((dst_pipe_x + 75) / 450, 2)
    # game_state[1] = dst_pipe_y #round((dst_pipe_y + 500) / 1000, 2)
    # game_state[2] = round(self.bird.y / 700, 2)
    # game_state[2] = self.bird.y_vel #round((self.bird.y_vel + 150) / 530, 2)
    
    
    return game_state #vector of size 20   
  
  
  #remember previous tuples to retrain whole network with them every time the game is over 
  def remember(self, current_state, action, reward, next_state, game_over):
    self.memory.append((current_state, action, reward, next_state, game_over)) # popleft if MAX_MEMORY is reached
    

  #after each gameover, train network on a whole batch 
  def train_batch(self) : 
    
      if len(self.memory) > Agent.BATCH_SIZE: batch = random.sample(self.memory, Agent.BATCH_SIZE) # list of tuples
      else: batch = self.memory
      
      states, actions, rewards, next_states, game_overs = zip(*batch)
      self.qtrainer.train(states, actions, rewards, next_states, game_overs)

      
  #save our qnetwork 
  def save_model(self, model_path) : 
    torch.save(self.qnetwork.state_dict(), model_path)
    print('model saved')
    
  #load and set model of our agent (from an existing trained model for instance)
  def set_model(self,model_path) : 
    self.qnetwork = QNet(input_size=4, output_size=2) 
    self.qnetwork.load_state_dict(torch.load(model_path))
    self.qtrainer = QTrainer(self.qnetwork, 1e-3, 0.9) #model, lr, gamma
      

