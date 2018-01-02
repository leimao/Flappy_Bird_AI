'''
Playing Flappy Bird Game using DQN_AI
Author: Lei Mao
Date: 4/20/2017 
'''

import cv2
import sys
sys.path.append("game/")
import os
import argparse
import shutil
import numpy as np
import wrapped_flappy_bird as game
from DQN_AI import DQN_AI

TEST_DIR = 'test/' # path for saving the test log

def Preprocess(observation):

    # Preprocess observation using cv2 to 72*40 gray image (raw image sizes ratio ~1.8)
    # Change the color of image to gray-scale
    observation = cv2.cvtColor(cv2.resize(observation, (72, 40)), cv2.COLOR_BGR2GRAY)
    # Change the color of image to binary (black or white)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    
    return observation

def Clear_PNGs(folder_path):
    
    # Clear all the png files in a folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            os.remove(folder_path + filename)
            
def Copy_PNGs(source_path, destination_path):

    # Copy all the png files from source folder to destination folder
    for filename in os.listdir(source_path):
        if filename.endswith('.png'):
            shutil.copy(source_path + filename, destination_path)

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
		
def TestFlappyBird(video_record = True):

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
    
    # Initialize video result log
    image_index = 0
    
    if video_record == True:
        if not os.path.exists(TEST_DIR + 'video/'):
            os.makedirs(TEST_DIR + 'video/')
        Clear_PNGs(folder_path = TEST_DIR + 'video/')
        cv2.imwrite(TEST_DIR + 'video/' + str(image_index) + '.png', cv2.cvtColor(cv2.transpose(observation), cv2.COLOR_BGR2RGB))
    
        if not os.path.exists(TEST_DIR + 'video_best/'):
            os.makedirs(TEST_DIR + 'video_best/')
        Clear_PNGs(folder_path = TEST_DIR + 'video_best/')
            
    observation = Preprocess(observation)
    AI_player.Current_State_Initialze(observation = observation)
    
    # Record score of each test
    test_round = 0
    score = 0
    score_highest = 0
    reward = 0
    terminal = False
    
    # Initialize test result log
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    fhand = open(TEST_DIR + 'test_log.txt', 'w')
    fhand.write('TEST_ROUND\tSCORE')
    fhand.write('\n')
    fhand.close()
		
    # AI starts playing
    while True:
        # Keep playing until hitting 'ctrl + c'
        # print('time step: %d' % AI_player.time_step)
        
        if reward == 1:
            score += 1
        if terminal == True:
            # Save test result log
            print('test round: %d, score: %d' % (test_round, score))
            fhand = open(TEST_DIR + 'test_log.txt', 'a')
            fhand.write(str(test_round) + '\t' + str(score))
            fhand.write('\n')
            fhand.close()
            # Check if this round of test is best
            if video_record == True:
                if score > score_highest:
                    score_highest = score
                
                    # Clear video_best folder
                    Clear_PNGs(folder_path = TEST_DIR + 'video_best/')
                    # Copy all the images in the video folder to video_best folder
                    Copy_PNGs(source_path = TEST_DIR + 'video/', destination_path = TEST_DIR + 'video_best/')
                    # Clear video folder
                    Clear_PNGs(folder_path = TEST_DIR + 'video/')                       

            # Reset score
            score = 0
            # Reset image index
            image_index = 0
            # Increase test_round value
            test_round += 1

        action = AI_player.AI_Action()
        next_observation, reward, terminal = flappybird.frame_step(action)
        
        # Save animated images
        if video_record == True:
            cv2.imwrite(TEST_DIR + 'video/' + str(image_index) + '.png', cv2.cvtColor(cv2.transpose(next_observation), cv2.COLOR_BGR2RGB))
            image_index += 1
        
        next_observation = Preprocess(next_observation)
        AI_player.Current_State_Update(observation = next_observation)

def TrainFlappyBirdResume():

    # Resume training in case of break 
    # Initialize Flappy Bird game
    flappybird = game.GameState()
    
    # Initialize AI for training
    num_actions = 2
    AI_player = DQN_AI(num_actions = num_actions, mode = 'train')
    
    # Set AI parameters to resume
    AI_player.Load_Model()
    AI_player.epsilon = 0 # user could adjust epsilon for the training after resume
    	
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
