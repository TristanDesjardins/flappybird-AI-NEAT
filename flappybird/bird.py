# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 21:29:13 2022

@author: X2029440
"""

from enum import Enum
import pygame 
import numpy as np 
from scipy.spatial import distance


class Color(Enum) :

  WHITE = (255, 255, 255)  # r,g,b
  BLACK = (0, 0, 0)
  
  RED = (255, 0, 0)
  BLUE = (0, 0, 255)
  
  LIGHT_RED = (255, 114, 118)
  LIGHT_BLUE = (0, 191, 255)
  
  GREEN = (41, 163, 41)
  DARK_GREEN = (0,110,0)
  YELLOW = (255, 204, 0)


#Class to define our Bird object 
class Bird : 
  
  
  
  def __init__(self, size=0.1, y=250, x=100) : 
    
    self.size = size #size of the bird (length of one side of the square)
    
    #position of the bird (center of the image)
    self.y = y #y position 
    self.x = x #x position

    #image bird 
    self.bird_image = pygame.image.load('bird.png')
    image_width, image_height = self.bird_image.get_size()
    self.bird_image = pygame.transform.scale(self.bird_image, \
                                            (self.size*image_width, self.size*image_height)) 
      
    self.rotated_image = self.bird_image.copy()
    
    self.bird_rect = self.bird_image.get_rect(center = (self.x, self.y))  #rect associated with the image
    
    self.y_vel = 0 #y velocity
    
    self.g = 100
    
    
  def jump(self, dt) :
    
    self.y_vel = -150
    self.y -= self.y_vel*dt
  
  
  def draw_bird(self, display) : 
    
    display.blit(self.rotated_image, self.bird_rect)
    
    
  #detect collision with pipe
  def is_colliding(self, pipe) : 
    return self.bird_rect.colliderect(pipe.bottom_pipe) \
           or self.bird_rect.colliderect(pipe.top_pipe) 
         
          
  #detect collision with ground or 'ceiling'
  def outside_borders(self) : 
    return self.y > 700 - self.size  or self.y < 0 
           
  
  #make brid fall because of gravity
  def fall(self, dt) : 

    self.y_vel += dt * self.g
    self.y += self.y_vel*dt

    self.bird_rect.center = (self.x, self.y)
    
  
  #make bird tilt down because of gravity 
  def tilt(self, dt) :  
    
    self.rotated_image = pygame.transform.rotate(self.bird_image, 
                                      np.math.degrees(np.arctan(-0.05*self.y_vel*dt)))
    
  
  #get state of the bird (input of our neural network for neat algorithm)
  #returns vector of size 3 : 
    # x distance to the next pipe 
    # y distance to the next pipe 
    # y velocity of the bird
  def get_game_state(self, game) : 
    
    game_state = [0,0,0]
    
    if game.next_pipe != None :
      dst_pipe_x = game.next_pipe.x - self.x
      dst_pipe_y = game.next_pipe.y - self.y
    
      game_state[0] = dst_pipe_x
      game_state[1] = dst_pipe_y
      
    game_state[2] = self.y_vel
    
    return game_state 
    
  

    
    
    
    
    
    
    
    
    
  
    
    
    
    
    
    
  