'''
Playing Flappy Bird Game using DQN_AI
Author: Lei Mao
Date: 4/20/2017 
'''

import cv2
import sys
import argparse
import numpy as np
sys.path.append("game/")
import wrapped_flappy_bird as game
from DQN_AI import DQN_AI

def Preprocess(observation):

    # Preprocess observation using cv2 to 72*40 gray image (raw image sizes ratio ~1.8)
    # Change the color of image to gray-scale
    observation = cv2.cvtColor(cv2.resize(observation, (72, 40)), cv2.COLOR_BGR2GRAY)
    # Change the color of image to binary (black or white)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    
    return observation

def TrainFlappyBird():

    # Train AI for the Flappy Bird game    
    # Initialize Flappy Bird game
    flappybird = game.GameState()
    
    # Initialize AI for training
    num_actions = 2
    AI_player = DQN_AI(num_actions = num_actions, mode = 'train')
	
    # AI training
    # Initialize the first state of AI with the first observation from the game
    action = np.array([1,0])  # idle
    observation, reward, terminal = flappybird.frame_step(action)
    observation = Preprocess(observation)
    AI_player.Current_State_Initialze(observation = observation)
    
    # AI starts training
    while True:
        # Keep training until hitting 'ctrl + c'
        print('time step: %d' % AI_player.time_step)
        action = AI_player.AI_Action()
        next_observation, reward, terminal = flappybird.frame_step(action)
        next_observation = Preprocess(next_observation)
        AI_player.Q_CNN_Train(action = action, reward = reward, observation = next_observation, terminal = terminal)
		
def TestFlappyBird():

    # Test AI for the Flappy Bird game
    # Initialize Flappy Bird game
    flappybird = game.GameState()
    
    # Load AI for the game
    num_actions = 2
    AI_player = DQN_AI(num_actions = num_actions, mode = 'test')
    AI_player.Load_Model()
	
    # AI starts to play the game
    # Initialize the first state of AI with the first observation from the game
    action = np.array([1,0])  # idle
    observation, reward, terminal = flappybird.frame_step(action)
    observation = Preprocess(observation)
    AI_player.Current_State_Initialze(observation = observation)
	
    # AI starts playing
    while True:
        # Keep playing until hitting 'ctrl + c'
        print('time step: %d' % AI_player.time_step)
        action = AI_player.AI_Action()
        next_observation, reward, terminal = flappybird.frame_step(action)
        next_observation = Preprocess(next_observation)
        AI_player.Current_State_Update(observation = observation)

def TrainFlappyBirdResume():

    # Resume training in case of break 
    # Initialize Flappy Bird game
    flappybird = game.GameState()
    
    # Initialize AI for training
    num_actions = 2
    AI_player = DQN_AI(num_actions = num_actions, mode = 'train')
    
    # Set AI parameters to resume
    AI_player.Load_Model()
    AI.epsilon = 0 # user could adjust epsilon for the training after resume
    	
    # AI training
    # Initialize the first state of AI with the first observation from the game
    action = np.array([1,0])  # idle
    observation, reward, terminal = flappybird.frame_step(action)
    observation = Preprocess(observation)
    AI_player.Current_State_Initialze(observation = observation)
    
    # AI starts training
    while True:
        # Keep training until hitting 'ctrl + c'
        print('time step: %d' % AI_player.time_step)
        action = AI_player.AI_Action()
        next_observation, reward, terminal = flappybird.frame_step(action)
        next_observation = Preprocess(next_observation)
        AI_player.Q_CNN_Train(action = action, reward = reward, observation = next_observation, terminal = terminal)

def main():

    parser = argparse.ArgumentParser(description = 'Designate AI mode')
    parser.add_argument('-m','--mode', help = 'train / test / resume', required = True)
    args = vars(parser.parse_args())

    if args['mode'] == 'train':
        TrainFlappyBird()
    elif args['mode'] == 'test':
        TestFlappyBird()
    elif args['mode'] == 'resume':
        TrainFlappyBirdResume()
    else:
        print('Please designate AI mode.')

if __name__ == '__main__':

    main()
