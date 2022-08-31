# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 21:51:11 2022

@author: X2029440
"""

import pygame 
from bird import Color 
import random 

#class to define one couple of pipes (bottom and top one)
class Pipe : 
  
  
  def __init__(self, width = 80, distance = 200) : 
    
    self.distance = distance #distance btw bottom and top pipe
    
    #coord of top left corner of bottom pipe
    self.x, self.y = 500, random.randint(100+self.distance,600) #random.randint(100+self.distance,600) 
    
    self.width = width #width of the pipes
    
    
    self.bottom_pipe = pygame.Rect(self.x, self.y, self.width, 700 - self.y)
    self.top_pipe = pygame.Rect(self.x, 0, self.width, self.y - self.distance)
    
    self.vel_x = 50
     
    
    
  def draw_pipe(self, display) : 
    
    pygame.draw.rect(display, Color.DARK_GREEN.value, self.bottom_pipe)
    pygame.draw.rect(display, Color.DARK_GREEN.value, self.top_pipe)
    
    
  def move_pipe(self, dt) : 
    
    self.x -= self.vel_x*dt
    self.bottom_pipe = pygame.Rect(self.x, self.y, self.width, 700 - self.y)
    self.top_pipe = pygame.Rect(self.x, 0, self.width, self.y - self.distance)
    
    

    
    
    
    
    