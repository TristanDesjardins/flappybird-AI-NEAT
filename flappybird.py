# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 21:31:50 2022

@author: X2029440
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 21:21:17 2022

@author: X2029440
"""

import pygame 
import sys 
from bird import Color 
from bird import Bird 
from pipe import Pipe  
import pickle 
import numpy as np 
import time 

#main class to play flappybird or let AI play 
class FlappyBird :
  
  
  BG_IMAGE = pygame.image.load("images/background.png")
  

  def __init__(self, window_width, window_height) : 
    
    self.window_width, self.window_height = window_width, window_height
    
    FlappyBird.BG_IMAGE = pygame.transform.scale(FlappyBird.BG_IMAGE, 
                                                 (self.window_width, self.window_height))
  
    self.bird = Bird()
    
    self.pipes = []
    self.next_pipe = None  #next coming pipe on the bird
    
    self.slow_down = True
    
    self.score = 0 
    
    self.game_started = False 
    
    self.score = 0
    self.best_score = 0 

    
    self.display = pygame.display.set_mode((self.window_width, self.window_height), 0, 32)
    pygame.init()
    pygame.display.set_caption('FLAPPYBIRD')
    
    
  def draw_background(self) : 
    self.display.blit(FlappyBird.BG_IMAGE, (0, 0))
    
    
  def reset_game(self) : 
    self.pipes = []
    self.bird = Bird()
    
    if self.score > self.best_score : 
      self.best_score = self.score
    self.score = 0 
      
    self.game_started = False 

    
    
    
  def draw_scores(self) : 
    
    pygame.font.init() 
    my_font = pygame.font.SysFont('Calibri', 20, bold=True)
    
    score = my_font.render('SCORE : ' + str(self.score), False, Color.BLACK.value)
    best_score = my_font.render('BEST SCORE : ' + str(self.best_score), False, Color.BLACK.value)
    
    self.display.blit(score, (0,0))
    self.display.blit(best_score, (0,20))
    
    
  #update game variables and ui at each frame 
  def update(self) : 
    
    self.draw_background()
    self.bird.draw_bird(self.display) 
    
    if self.game_started : 
      
      self.bird.fall(0.1)
      self.bird.tilt(0.1)
    
    
      for pipe in self.pipes : 
           
        pipe.draw_pipe(self.display)
        pipe.move_pipe(0.1)
        
        if self.bird.x == pipe.x + 130  : 
          self.next_pipe = self.pipes[self.pipes.index(self.next_pipe)+1]
          self.score += 1
        
        if self.bird.is_colliding(pipe) :
          print('collision!')
          self.reset_game()
        
      if self.bird.outside_borders() : self.reset_game()
        
      
        
    if self.next_pipe != None and self.game_started: #draw red line (game state)
      pygame.draw.line(self.display, 
                       Color.RED.value,
                       (self.bird.x, self.bird.y), 
                       (self.next_pipe.x, self.next_pipe.y))
       
    self.draw_scores()
  
    pygame.display.update()
    
    
    
  #create new pipes and pop older ones
  def handle_pipes(self) : 
    if self.game_started : 
      if len(self.pipes) == 0 : 
        self.pipes.append(Pipe())
        self.next_pipe = self.pipes[0]
      
      elif len(self.pipes) < 4 : 
        pipe = Pipe()
        pipe.x = self.pipes[-1].x + 350 #distance btw 2 pipes 
        self.pipes.append(pipe)
        
      if self.pipes[0].x < -100 : 
        self.pipes.pop(0)



  #if you want to run the game and play yourself
  def run_game(self):

    next_update = 0  #time of next update 
    dt_updates = 1/60 #time btw two updates

   
    first = True
    
    while True : #game loop 

      current_time = pygame.time.get_ticks()/1000 #in seconds 
      
      #update ui 
      if  current_time >= next_update :  #update game only every 'FPS' seconds
        next_update = current_time + dt_updates
        
        self.update()
        
        #pipes
        self.handle_pipes()     
        
      #all event should be treated inside this unique for loop (otherwise really slow)
      for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_SPACE:
            self.game_started = True 
            self.bird.jump(0.1)
          
        if event.type == pygame.QUIT:
          pygame.quit()
          sys.exit()
          
          
          
  #run game from trained model (to train model, use flappybirdai_neat.py)   
  #use mousewheel to speed up process     
  def run_from_AI(self, model_path = 'model.pkl') :
    
    with open(model_path, 'rb') as f:
      model = pickle.load(f)
      
    next_update = 0  #time of next update 
    dt_updates = 1/np.inf #time btw two updates #1/np.inf
    
    first = True
    
    while True : #game loop 
    
      self.game_started = True 

      current_time = pygame.time.get_ticks()/1000 #in seconds 
      
      #update ui 
      if  current_time >= next_update :  #update game only every 'FPS' seconds
        next_update = current_time + dt_updates

        self.update()
        
        game_state = self.bird.get_game_state(self)
        output = model.activate(game_state) #output neural net attached to each bird
        
        #we use a tanh activation function so result will be between -1 and 1. if over 0.5 jump
        if output[0] > 0.5 : 
          self.bird.jump(0.1)
         
        self.handle_pipes()
              
      #slow down AI with mousewheel events
      for event in pygame.event.get():
        if event.type == pygame.MOUSEWHEEL:
          if event.y == 1 : self.slow_down = False 
          if event.y == -1 : self.slow_down = True
        if event.type == pygame.QUIT:
          pygame.quit()
          sys.exit()
          
      if self.slow_down : time.sleep(0.012)
        
          
        
          
if __name__ == "__main__":
  
  window_width = 500
  window_height = 700
  
  #if true, AI plays (from previously trained model), 
  #if False, YOU play! Enjoy!
  AI = True
    
  game = FlappyBird(window_width, window_height)
  
  if AI : 
    game.run_from_AI(model_path = 'best_model.pkl')
  else : 
    game.run_game()
        
        
        
        
        
        
        
        
        
        
        
    