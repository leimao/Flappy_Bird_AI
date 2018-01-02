'''
Deep Q-Learning AI Game Player
Author: Lei Mao
Date: 4/19/2017
Introduction: 
The DQN_AI used Deep Q-Learning to study optimal solutions to play a certain game, assuming that the game is a Markov Decision Process. For the training step, the game API exports the game state (game snapshot), reward of the game state, and the signal of game termination to the DQN_AI for learning. For the test step, the DQN_AI only takes the game state as input and output operations that the DQN_AI thinks optimal. 
Features:
The DQN_AI was written in python using Tensorflow and Keras API, which is extremely neat and simple to understand. The game for illustration here is 'FlappyBird'. There are exsiting 'FlappyBird' python APIs on the internet. Those APIs were adapted and slightly modified to meet the author's personal perferences.
'''

import tensorflow as tf
import numpy as np
import keras
import random
import os
import keras
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

# Hyperparameters
IMG_ROWS = 40 # number of image rows as input 
IMG_COLS = 72 # number of image columns as input 
GAME_STATE_FRAMES = 4 # number of images stacked as input
GAMMA = 0.99 # decay rate of past observations
EPSILON_INITIALIZED = 1.0 # probability epsilon used to determine random actions
EPSILON_FINAL = 0 # final epsilon after decay
BATCH_SIZE = 32 # number of sample size in one minibatch
LEARNING_RATE = 0.00001 # learning rate in deep learning
FRAME_PER_ACTION = 1 # number of frames per action
REPLAYS_SIZE = 200 # maximum number of replays in cache
TRAINING_DELAY = 2000 # time steps before starting training for the purpose of collecting sufficient replays to initialize training
EXPLORATION_TIME = 50000 # time steps used during training before changing the epsilon
SAVING_PERIOD = 5000 # period of time steps to save the model
LOG_PERIOD = 100 # period of time steps to save the log of training
MODEL_DIR = 'model/' # path for saving the model
LOG_DIR = 'log/' # path for saving the training log


class DQN_AI():

    def __init__(self, num_actions, mode):
    
        # Initialize the number of player actions available in the game
        self.num_actions = num_actions
        # Determine the shape of input to the model
        self.input_shape = [IMG_ROWS, IMG_COLS, GAME_STATE_FRAMES]
        # Initialize the model
        self.model = self.Q_CNN_Setup()
        # Initialize game_replays used for caching game replays
        self.game_replays = deque()
        # Initialize time_step to count the time steps during training
        self.time_step = 0
        # Initialize the mode of AI
        self.mode = mode
        # Initialize epsilon which controls the probability of choosing random actions
        self.epsilon = 0
        if self.mode == 'train':
            self.epsilon = EPSILON_INITIALIZED
        elif self.mode == 'test':
            self.epsilon = 0
        else:
            raise('AI mode error.')
        
    def Current_State_Initialze(self, observation):
    
        # Initialize current_state with the observation from the game
        self.current_state = np.stack(tuple([observation.tolist()] * GAME_STATE_FRAMES), axis = 2)

    def Current_State_Update(self, observation):
    
        # Update current_state with the observation from the game
        self.current_state = np.append(self.current_state[:,:,1:], observation.reshape((self.input_shape[0],self.input_shape[1],1)), axis = 2)
        self.time_step += 1
        
    def State_Format(self, data):
        
        # Add the fourth dimension to the data for Tensorflow
        # For example a single data with dimension of [80,80,4] to [1,80,80,4]
        return data.reshape(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        
        
    def Q_CNN_Setup(self):
    
        # Prepare Convolutional Neural Network for Q-Learning
        # Note that we did not use regularization here
        model = Sequential()
        # CNN layer_1
        model.add(Conv2D(16, kernel_size = (8, 8), activation = 'relu', input_shape = self.input_shape))
        # Max pooling
        model.add(MaxPooling2D(pool_size = (2, 2)))
        # CNN layer_2
        model.add(Conv2D(32, (4, 4), activation = 'relu'))
        # CNN layer_3
        model.add(Conv2D(32, (3, 3), activation = 'relu'))
        # Flatten data to 1D
        model.add(Flatten())
        # FC layer_1
        model.add(Dense(256, activation = 'relu'))
        # FC layer_2
        model.add(Dense(self.num_actions))
        # Optimizer
        optimizer = keras.optimizers.Adam(lr = LEARNING_RATE)
        # Compile the model
        model.compile(loss = keras.losses.mean_squared_error, optimizer = optimizer)
        
        return model
        
    def Q_CNN_Train_Batch(self, minibatch):
    
        # Elements in minibatch consists tuple (current_state, state_action, state_reward, next_state, terminal)        
        # Generate inputs and targets from minibatch data and model
        
        # Initialize inputs
        inputs = np.zeros((len(minibatch), self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        targets = np.zeros((len(minibatch), self.num_actions))
        
        # Prepare inputs and calculate targets
        for i in xrange(len(minibatch)):
        
            current_state = minibatch[i][0]
            state_action = minibatch[i][1]
            state_reward = minibatch[i][2]
            next_state = minibatch[i][3]
            terminal = minibatch[i][4]
            
            Qs_current_state = self.model.predict(self.State_Format(current_state))[0]
            Qs_next_state = self.model.predict(self.State_Format(next_state))[0]
            
            inputs[i] = current_state
            targets[i] = Qs_current_state
            
            if terminal:
                targets[i,np.argmax(state_action)] = state_reward
            else:
                targets[i,np.argmax(state_action)] = state_reward + GAMMA * np.max(Qs_next_state)
        
        # Train on batch
        loss = self.model.train_on_batch(inputs, targets)
        
        # print('loss: %f' %loss)
        
        # Return training details for print
        return loss, Qs_current_state.astype(np.float), targets[-1].astype(np.float)
        
    def AI_Action(self):
    
        # AI calculate optimal actions for the current state
        state_action = np.zeros(self.num_actions)
        
        if self.time_step % FRAME_PER_ACTION == 0:           
            if random.random() < self.epsilon:
                # Choose random action
                #action_index = random.randint(0,self.num_actions-1)
                #state_action[action_index] = 1
                # Choose even psudorandom action specific for Flappy Bird game
                if random.random() <= 1./5:
                    action_index = 1
                    state_action[action_index] = 1
                else:
                    action_index = 0
                    state_action[action_index] = 1
                              
            else:
                # Choose the optimal action from the model
                Qs = self.model.predict(self.State_Format(self.current_state))[0]
                action_index = np.argmax(Qs)
                state_action[action_index] = 1
        else:
            action_index = 0
            state_action[action_index] = 1
            
        # Update epsilon
        if (self.mode == 'train') and (self.epsilon > 0):
            if (self.time_step >= TRAINING_DELAY) and (self.time_step < (TRAINING_DELAY + EXPLORATION_TIME)):
                self.Epsilon_Update()
            
        return state_action
    
    def Epsilon_Update(self):
        
        # Update epsilon during training
	    self.epsilon -= (EPSILON_INITIALIZED - EPSILON_FINAL)/EXPLORATION_TIME
        
    def Q_CNN_Train(self, action, reward, observation, terminal):
    
        # Next state after taking action at current state
        next_state = np.append(self.current_state[:,:,1:], observation.reshape((self.input_shape[0],self.input_shape[1],1)), axis = 2)
        
        # Add the replay to game_replays
        self.game_replays.append((self.current_state, action, reward, next_state, terminal))
        
        # Check game_replays exceeds the size specified
        if len(self.game_replays) > REPLAYS_SIZE:
            # Remove the oldest replay
            self.game_replays.popleft()
        
        # Start training after training delay
        loss = 'NA'
        Qs_predicted_example = 'NA'
        Qs_target_example = 'NA'
        if self.time_step > TRAINING_DELAY:
            # Train Q_CNN on batch
            minibatch = random.sample(self.game_replays, BATCH_SIZE)
            loss, Qs_predicted_example, Qs_target_example = self.Q_CNN_Train_Batch(minibatch = minibatch)
            
        # Save model routinely
        if self.time_step % SAVING_PERIOD == 0:
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
            self.model.save(MODEL_DIR + 'AI_model.h5')

        # Print Training Information
        if self.time_step < TRAINING_DELAY:
            stage = 'DELAY'
        elif (self.time_step >= TRAINING_DELAY) and (self.time_step < (TRAINING_DELAY + EXPLORATION_TIME)):
            stage = 'EXPLORATION'
        else:
            stage = 'TRAINING'
            
        print('TIME_STEP', self.time_step, '/ STAGE', stage, '/ EPSILON', self.epsilon, '/ ACTION', np.argmax(action), '/ REWARD', reward, '/ Qs_PREDICTED_EXAMPLE', Qs_predicted_example, '/ Qs_TARGET_EXAMPLE', Qs_target_example, '/ Loss', loss)
        
        # Save training log routinely
        if self.time_step == 0:
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            # Create training log file
            fhand = open(LOG_DIR + 'training_log.txt', 'w')
            fhand.write('TIME_STEP\tSTAGE\tEPSILON\tACTION\tREWARD\tQs_PREDICTED_EXAMPLE\tQs_TARGET_EXAMPLE\tLoss')
            fhand.write('\n')
            fhand.close()
            # Create training parameters file
            fhand = open(LOG_DIR + 'training_parameters.txt', 'w')
            fhand.write('IMG_ROWS\t' + str(IMG_ROWS) + '\n')
            fhand.write('IMG_COLS\t' + str(IMG_COLS) + '\n')
            fhand.write('GAME_STATE_FRAMES\t' + str(GAME_STATE_FRAMES) + '\n')
            fhand.write('GAMMA\t' + str(GAMMA) + '\n')
            fhand.write('EPSILON_INITIALIZED\t' + str(EPSILON_INITIALIZED) + '\n')
            fhand.write('EPSILON_FINAL\t' + str(EPSILON_FINAL) + '\n')
            fhand.write('BATCH_SIZE\t' + str(BATCH_SIZE) + '\n')
            fhand.write('LEARNING_RATE\t' + str(LEARNING_RATE) + '\n')
            fhand.write('FRAME_PER_ACTION\t' + str(FRAME_PER_ACTION) + '\n')
            fhand.write('REPLAYS_SIZE\t' + str(REPLAYS_SIZE) + '\n')
            fhand.write('TRAINING_DELAY\t' + str(TRAINING_DELAY) + '\n')
            fhand.write('EXPLORATION_TIME\t' + str(EXPLORATION_TIME) + '\n')
            fhand.write('SAVING_PERIOD\t' + str(SAVING_PERIOD) + '\n')
            fhand.write('LOG_PERIOD\t' + str(LOG_PERIOD) + '\n')
            fhand.close()

        if self.time_step % LOG_PERIOD == 0:
            fhand = open(LOG_DIR + 'training_log.txt', 'a')
            fhand.write(str(self.time_step) + '\t' + str(stage) + '\t' + str(self.epsilon) + '\t' + str(np.argmax(action)) + '\t' + str(reward) + '\t' + str(Qs_predicted_example) + '\t' + str(Qs_target_example) + '\t' + str(loss))
            fhand.write('\n')
            fhand.close()

        # Update current state
        self.current_state = next_state
        
        # Increase time step
        self.time_step += 1
        
    def Load_Model(self):
    
        # Load the saved model
        self.model = load_model(MODEL_DIR + 'AI_model.h5')
        




            
            
        
            
            
            



