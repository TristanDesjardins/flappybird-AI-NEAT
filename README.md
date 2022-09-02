# FlappyBird AI - NEAT genetic algorithm 
FlappyBird AI using 'Neuroevolution of augmenting topologies' aka NEAT (with neat library and pygame) <br/> 
Project done as a challenge in my free time :blush: <br/> 
Thank's CodeBullet for the idea! 


## Preview 
**training phase**             |  **After training**
:-------------------------:|:-------------------------:
![flap_training](https://user-images.githubusercontent.com/62900180/188199607-8eb74cd4-dc56-4ad5-988e-9757f5c2bc22.gif)| ![flap_trained](https://user-images.githubusercontent.com/62900180/188218028-7f185322-de87-4fb6-a018-be91f73529ac.gif)


## General Idea 

Here's how I adapted NEAT for my case:

- We generate N birds and associate them N randomly initialized neural networks 
- We make them all play 
- fitness is essentially how much time the bird survived

We can then take the best birds (birds with most fitness), mutate them, and start over with a new population of birds.
We repeat this process (called a generation) until birds are good enough.

Check NEAT algo for more info.



## Main files
- flappybird_neat.py : to run NEAT algo and save best model 
- flappybird.py : if 'AI' == True, AI will be playing. if 'AI' == False, you can play the game yourself 

## Installation 
- Python 3.9.12
- [pygame 2.1.2](https://www.pygame.org/news) : for the game engine 
- [neat 0.4.1](https://neat-python.readthedocs.io/en/latest/installation.html) : for the AI 
