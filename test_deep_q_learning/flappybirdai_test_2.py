# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:16:08 2022

@author: X2029440
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 19:37:42 2022

@author: X2029440
"""


import pygame 
import sys 
from bird import Color 
from bird import Bird 
from pipe import Pipe 
from agent import Agent 
import time 
import numpy as np 
import datetime 
from scipy.spatial import distance


class FlappyBird :
  
  
  BG_IMAGE = pygame.image.load("background.png")
  

  def __init__(self, window_width, window_height) : 
    
    self.window_width, self.window_height = window_width, window_height
    
    FlappyBird.BG_IMAGE = pygame.transform.scale(FlappyBird.BG_IMAGE, 
                                                 (self.window_width, self.window_height))
  
    bird = Bird()
    self.agent = Agent(bird)
    self.nb_games = 0 
    
    self.next_pipe = None 
    
    self.slow_down = False 
    
    self.pipes = []
    self.next_pipe = None  #next coming pipe on the bird
    
    self.succeed = False 
    
    self.score = 0 
    
    self.score = 0
    self.best_score = 0 

    
    self.display = pygame.display.set_mode((self.window_width, self.window_height), 0, 32)
    pygame.init()
    pygame.display.set_caption('FLAPPYBIRD')
    
    
  def draw_background(self) : 
    self.display.blit(FlappyBird.BG_IMAGE, (0, 0))
    
    
  def reset_game(self) : 
    self.pipes = []
    self.agent.bird  = Bird()
    
    if self.score > self.best_score : 
      self.best_score = self.score
    self.score = 0 
      
    
    
  def draw_scores(self) : 
    
    pygame.font.init() 
    my_font = pygame.font.SysFont('Calibri', 20, bold=True)
    
    score = my_font.render('SCORE : ' + str(self.score), False, Color.BLACK.value)
    best_score = my_font.render('BEST SCORE : ' + str(self.best_score), False, Color.BLACK.value)
    
    self.display.blit(score, (0,0))
    self.display.blit(best_score, (0,20))
    
    
    
  def update(self) : 
    
    reward, game_over = 0, False 
    
    self.draw_background()
    self.agent.bird.draw_bird(self.display) 
    

    self.agent.bird.fall(0.1)
    self.agent.bird.tilt(0.1)
  
  
    for pipe in self.pipes : 
      pipe.draw_pipe(self.display)
      pipe.move_pipe(0.1)
      

      if self.agent.bird.x == pipe.x + 80 : 
        self.succeed = True 
        self.score += 1 
        self.next_pipe = self.pipes[self.pipes.index(self.next_pipe)+1]

      
      if self.agent.bird.is_colliding(pipe) :
        game_over = True
        self.reset_game()
      
      if self.agent.bird.outside_borders() : 
        game_over = True
        self.reset_game()
        
    self.draw_scores()
  
    pygame.display.update()
    
    return reward, game_over
  


  #game loop 
  def train_agent(self):
    
    next_update = 0  #time of next update 
    dt_updates = 1/60 #time btw two updates

    first = True
    nb_loops = 0 
    last_jump = 0 
    
    game_over = False 
    
    last_pipe = []
    
    
    try : 
      
      while True : #game loop 
      
        train = True #whether should train or not 
      
        current_time = pygame.time.get_ticks()/1000 #in seconds
        
        #update ui 
        if  current_time >= next_update or not self.slow_down :  #update game only every 'FPS' seconds

          seconds = int(pygame.time.get_ticks()/1000)
          # print("AIs have been training for:", datetime.timedelta(seconds=seconds))
          # print("AIs have played", self.nb_games,'games') 
        
          next_update = current_time + dt_updates
          
          if nb_loops >= 1 : 
            current_state = self.agent.get_game_state(self)
            # print('state:', current_state)
            action = self.agent.get_action(current_state, self.nb_games)
            if np.argmax(action) == 0 :
              if nb_loops - last_jump > 20 : #min nb loops between two jumps
                last_jump = nb_loops
                self.agent.bird.jump(0.1)
              else : train = False #if the action was to jump but we didn't (shouldn't train qnetwork from that)
            
          
          reward, game_over = self.update()
          
          if nb_loops >= 1 : 
            if train :
              next_state = self.agent.get_game_state(self)
              last_pipe.append((current_state, action, reward, next_state, game_over))
              self.agent.remember(current_state, action, reward, next_state, game_over)  
              
              
          if (self.succeed or game_over) and len(last_pipe) > 0: 

            x,y = last_pipe[-1][0][0], last_pipe[-1][0][1] - 100 #top left corner of pipe (pipe - bird)
            reward = - np.sqrt(x**2 + y**2)
            
            print(reward)
            self.succeed = False 
            states, actions, rewards, next_states, game_overs = zip(*last_pipe)
            rewards = (reward,) * len(rewards)
            self.agent.qtrainer.train(states, actions, rewards, next_states, game_overs) #train
            
            last_pipe = []
            
          
          #pipes
          if len(self.pipes) == 0 : 
            self.pipes.append(Pipe())
            self.next_pipe = self.pipes[0]
          
          elif len(self.pipes) < 4 : 
            pipe = Pipe()
            pipe.x = self.pipes[-1].x + 350 #distance btw 2 pipes 
            self.pipes.append(pipe)
            
          if self.pipes[0].x < -100 : 
            self.pipes.pop(0)
            
          if game_over : #experience replay 
            self.nb_games += 1 
            # self.agent.train_batch()
            
            
          nb_loops += 1 
            
          
        #slow down learning with mouse wheel events
        for event in pygame.event.get():
          if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
              self.agent.bird.jump(0.1)
          if event.type == pygame.MOUSEWHEEL:
            if event.y == 1 : self.slow_down = False 
            if event.y == -1 : self.slow_down = True
          if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            
    finally : 
      self.agent.save_model('model.pth')
      
      
      
      
  
  def play_from_model(self) : 
    
    self.agent.set_model('model.pth')
    
    next_update = 0  #time of next update 
    dt_updates = 1/60 #time btw two updates

    first = True
    nb_loops = 0 
    last_jump = 0 
    

    while True : #game loop 
    
      train = True #whether should train or not 
    
      current_time = pygame.time.get_ticks()/1000 #in seconds

      #update ui 
      if  current_time >= next_update or not self.slow_down :  #update game only every 'FPS' seconds
        next_update = current_time + dt_updates
        
        nb_loops += 1 
        
        self.update()
        game_state = self.agent.get_game_state(self)
        action = self.agent.get_action(game_state, self.nb_games)
        if np.argmax(action) == 0 : 
          self.agent.bird.jump(0.1)
        
        
        #pipes
        if len(self.pipes) == 0 : self.pipes.append(Pipe())
        
        elif len(self.pipes) < 4 : 
          pipe = Pipe()
          pipe.x = self.pipes[-1].x + 350 #distance btw 2 pipes 
          self.pipes.append(pipe)
          
        if self.pipes[0].x < -100 : 
          self.pipes.pop(0)

        for event in pygame.event.get():  
          if event.type == pygame.MOUSEWHEEL:
            if event.y == 1 : self.slow_down = False 
            if event.y == -1 : self.slow_down = True
          if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    

          
        

if __name__ == "__main__":
  
  window_width = 500
  window_height = 700
  
  
  game = FlappyBird(window_width, window_height)
  # game.play_from_model()
  
  game.train_agent()
  
        
        
        
        
        
        
        
        
        
        
        
        
    