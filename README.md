# flappybird-AI-NEAT
FlappyBird AI using 'Neuroevolution of augmenting topologies' aka NEAT (with neat library and pygame) <br/> 
Project done as a challenge in my free time :blush: <br/> 
Thank's CodeBullet for the idea! 

## General Idea 

Here's how I adapted NEAT for my case:

- We generate N birds and associate them N randomly initialized neural networks 
- We make them all play 
- fitness is essentially how much time the bird survived

We can then take the best birds (birds with most fitness), mutate them, and start over with a new population of birds.
We repeat this process (called a generation) until birds are good enough.

Check NEAT algo for more info.

## Preview 

Just run 'flappybird_neat.py' file to see our birds in action! Here's our birds after 11 generations:
<br/>
<img src="https://user-images.githubusercontent.com/62900180/187653342-3a4e1fa2-c674-4d40-a570-c8ad9941b350.gif" height="500">
<br/>

Here's the best model obtained with NEAT. 14000, pretty good right? 
<br/>
<img src="https://user-images.githubusercontent.com/62900180/187653442-7166b03b-6b17-4e29-a33f-8f50897327cf.gif" height="500">
<br/>

## Main files
- flappybird_neat.py : to run NEAT algo and saved best model 
- flappybird.py : if 'AI' == True, AI will be playing. if 'AI' == False, you can play the game yourself! 

## Installation 
- python 3.9.12
