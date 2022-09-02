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
training phase             |  After training
:-------------------------:|:-------------------------:
![flap_training](https://user-images.githubusercontent.com/62900180/188199607-8eb74cd4-dc56-4ad5-988e-9757f5c2bc22.gif)| ![flap_trained](https://user-images.githubusercontent.com/62900180/188199583-87ab7b9d-f616-4a99-99f3-cf91bced2172.gif)

<br/>


## Main files
- flappybird_neat.py : to run NEAT algo and saved best model 
- flappybird.py : if 'AI' == True, AI will be playing. if 'AI' == False, you can play the game yourself! 

## Installation 
- python 3.9.12
