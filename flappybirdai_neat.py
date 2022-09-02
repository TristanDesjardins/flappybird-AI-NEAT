# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:18:15 2022

@author: X2029440
"""

import neat
import pygame 
import sys 
from bird import Color 
from bird import Bird 
from pipe import Pipe  
import random 
import time 
import numpy as np 
import datetime
import pickle



#Class to 'train' AIs with neat algorithm
#possible improvements : 
  # normalize input's of our networks
  # add x and y distance to top pipe (not only bottom pipe) -> it shoudn't add info because the gap between
  # the two pipes is constant but it may help the network learn faster
class FlappyBird :
  
  
  BG_IMAGE = pygame.image.load("images/background.png")
  

  def __init__(self, window_width, window_height) : 
    
    self.window_width, self.window_height = window_width, window_height
    
    FlappyBird.BG_IMAGE = pygame.transform.scale(FlappyBird.BG_IMAGE, 
                                                 (self.window_width, self.window_height))
    
    self.slow_down = False #slow down process during training 
  
    self.nb_gen = 0 
    
    self.birds = []
    
    self.pipes = []
    self.next_pipe = None  #next coming pipe on the bird
    
    self.score = 0 
    
    self.score = 0
    self.best_score = 0 

    
    self.display = pygame.display.set_mode((self.window_width, self.window_height), 0, 32)
    pygame.init()
    pygame.display.set_caption('FLAPPYBIRD')
    
    
  def draw_background(self) : 
    self.display.blit(FlappyBird.BG_IMAGE, (0, 0))
    

  def draw_scores(self) : 
    
    pygame.font.init() 
    my_font = pygame.font.SysFont('Calibri', 20, bold=True)
    
    score = my_font.render('SCORE : ' + str(self.score), False, Color.BLACK.value)
    best_score = my_font.render('BEST SCORE : ' + str(self.best_score), False, Color.BLACK.value)
    pop_size = my_font.render('POP SIZE : ' + str(len(self.birds)), False, Color.BLACK.value)
    nb_gen = my_font.render('GEN : ' + str(self.nb_gen), False, Color.BLACK.value)
    
    self.display.blit(score, (0,0))
    self.display.blit(best_score, (0,20))
    self.display.blit(pop_size, (0,50))
    self.display.blit(nb_gen, (0,70))
    
    
          
  #update function for training with neat 
  def update_neat(self, nets, ge) : 
    
    self.draw_background()
    
    score_update = False #update score only once (not for every bird that goes past a pipe)
      
    idx = list(range(len(self.birds))) #indices of non dead birds (initially, all index are there)

    #we go through list backward in order to delete element
    for i, bird in reversed(list(enumerate(self.birds))): 

      ge[i].fitness += 0.1 #add fitness every frame 
      bird.draw_bird(self.display) 

        
      bird.fall(0.1)
      bird.tilt(0.1)
      
      output = nets[i].activate(bird.get_game_state(self)) #output neural net attached to each bird
      
      # we use a tanh activation function so result will be between -1 and 1. if over 0.5 jump
      if output[0] > 0.5 : 
        bird.jump(0.1)
      
      #draw red line (game state) for each bird 
      if self.next_pipe != None : 
        pygame.draw.line(self.display, 
                         Color.RED.value,
                         (bird.x, bird.y), 
                         (self.next_pipe.x, self.next_pipe.y))
        
      #check for collision with next pipe 
      if self.next_pipe != None : 
        if bird.is_colliding(self.next_pipe) or bird.outside_borders() :
          self.birds.remove(bird) #we remove dead birds
          ge[i].fitness -= 1 #if collision, we remove some fitness
          idx.pop(i) #remove index of dead birds
          
      
        #update score + next pipe 
        if bird.x == self.next_pipe.x + 130 and self.next_pipe and not score_update : 
          score_update = True 
          self.next_pipe = self.pipes[self.pipes.index(self.next_pipe)+1]
          
          #update score and best score
          self.score += 1
          if self.score > self.best_score :  
            self.best_score = self.score 
            
          #if score is big enough we can save a model 
          if self.score > 1000 : 
            with open('model.pkl', 'wb') as f:
                pickle.dump(nets[i], f)
                print('model saved')
            
          
    #we remove networks and genomes of dead birds
    nets = [nets[i] for i in idx]
    ge = [ge[i] for i in idx]
          
    #update pipes
    for pipe in self.pipes : 
         
      pipe.draw_pipe(self.display)
      pipe.move_pipe(0.1)

    self.draw_scores()
  
    pygame.display.update()

    return nets, ge
          
  
  #this function loops everytime it comes to its end 
  #every loop, new population (new genomes and networks) is made out of the best birds (birds with best fitness)
  #here i defined the fitness as how far the bird went (fitness+=1 at every frame)
  def eval_genomes(self, genomes, config) : 
    
    #nets : neural networks attached to each bird 
    #ge : genome object attached to each bird (for fitness) -> fitness is a score that determines how good a bird did (in our case, how far it went) 
    self.score = 0
    nets, ge = [], [] 
    self.birds = []
    self.pipes = []

    for genome_id, genome in genomes : 
      genome.fitness = 0 #initialize fitness 
      net = neat.nn.FeedForwardNetwork.create(genome, config)
      nets.append(net)
      ge.append(genome)
      self.birds.append(Bird())
      
      
    next_update = 0  #time of next update 
    dt_updates = 1/np.inf #time btw two updates

    first = True
    
    while len(self.birds) > 0 : #game loop 

      current_time = pygame.time.get_ticks()/1000 #in seconds 
      
      #update ui 
      if  current_time >= next_update :  #update game only every 'FPS' seconds
      
        seconds = int(pygame.time.get_ticks()/1000)
        print("AIs have been training for:", datetime.timedelta(seconds=seconds))
      
        next_update = current_time + dt_updates
        
        nets, ge = self.update_neat(nets, ge) #upate game + update networks and genomes (remove those of dead birds)
        
        #pipes (add new and remove old) 
        if len(self.pipes) == 0 : 
          self.pipes.append(Pipe())
          self.next_pipe = self.pipes[0]
        
        elif len(self.pipes) < 4 : 
          pipe = Pipe()
          pipe.x = self.pipes[-1].x + 350 #distance btw 2 pipes 
          self.pipes.append(pipe)
          
        if self.pipes[0].x < -100 : 
          self.pipes.pop(0)
            
 
      #all event should be treated inside this unique for loop (otherwise really slow)
      for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_SPACE:
            for bird in self.birds : bird.jump(0.1) 
          
        if event.type == pygame.QUIT:
          pygame.quit()
          sys.exit()
        if event.type == pygame.MOUSEWHEEL:
          if event.y == 1 : self.slow_down = False 
          if event.y == -1 : self.slow_down = True
          
      if self.slow_down : 
        time.sleep(0.007)
    
    self.nb_gen += 1 
    print('done')

      
    
  #to run neat algo using eval_genomes function in which we define what to do during one generation
  def run(self, config_file, nb_gens):
    
      """
      runs the NEAT algorithm to train a neural network to play flappy bird.
      :param config_file: location of config file
      :return: None
      """
      config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_file)
  
      # Create the population, which is the top-level object for a NEAT run.
      p = neat.Population(config)
  
      # Add a stdout reporter to show progress in the terminal.
      p.add_reporter(neat.StdOutReporter(True))
      stats = neat.StatisticsReporter()
      p.add_reporter(stats)
      #p.add_reporter(neat.Checkpointer(5))
  
      winner = p.run(self.eval_genomes, nb_gens) #eval_genomes(genomes, config) function where we treat our genomes (doesn't return anythign)
  
      # show final stats
      print('\nBest genome:\n{!s}'.format(winner))
      
      #save best bird's network
      winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
      with open('model.pkl', 'wb') as f:
          pickle.dump(winner_net, f)
      
      
  

          
if __name__ == "__main__":
  
  window_width = 500
  window_height = 700
  
  game = FlappyBird(window_width, window_height)
  
  nb_gens = 50 #number of generations 
  config_path = "config-feedforward.txt"
  
  #train AIs with neat algorithm
  #you can slow down process using mousewheel 
  game.run(config_path, nb_gens)

  


