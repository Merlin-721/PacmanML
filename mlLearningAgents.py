# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util
import numpy as np
import copy

class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.8, gamma=0.8, numTraining = 2000):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        self.QTable = np.empty((0,3))
        self.moves = []
        self.scoreLast = None
        self.S = None
        self.action = None
        self.dirDic = {
            0:Directions.NORTH,
            1:Directions.SOUTH,
            2:Directions.EAST,
            3:Directions.WEST
        }

    def registerInitialState(self,state):
        self.scoreLast = state.getScore()

    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    def setEpsilon(self, value):
        self.epsilon = value

    def setAlpha(self, value):
        self.alpha = value

    def numToDir(self,num):
        '''
        Converts a number to a direction.

        Args: num - a number
        Returns: a direction, eg. Directions.SOUTH
        '''
        return self.dirDic[num]

    def dirToNum(self,direction):
        '''
        Converts a direction to a number.

        Args: direction - a direction
        Returns: a number
        '''
        return self.dirDic.values().index(direction)

    def addState(self,env):
        '''
        Adds a state to global table and initialises Q values and visits to 0

        Args: env - state representing food, ghost and pacman positions
        '''
        num = 4
        #[(state),[Q values], [visits]]
        row = np.array([[env,np.zeros(num),np.zeros(num)]])
        self.QTable = np.append(self.QTable,row, axis=0)
        
    def getGreedy(self,S):
        '''
        Gets action with highest Q-value given state, weighted
        by number of visits.

        Args: S - index of state in Q-table
        Returns: x - action number
        '''
        # Q's * N's
        product = self.QTable[S][1] * self.QTable[S][2]
        sort = reversed(np.argsort(product))
        for x in sort:
            if x in self.moves: return x
        raise Exception("Chosen action not legal")

    def QAction(self,env):
        ''' 
        Checks if state has been seen before by checking Q-table, 
        and returns actions based on Q-values if present.
        If not present it will add the state to Q-table.

        Args: env - current state
        Returns: (random, S) - (random action, index of state)
                                if we have seen state before and 
                                random value is below epsilon
                (greedy, S) - (highest Q-value action, index of state)
                                if random value above epsilon
                (random, index) - (random action, index of state)
                                if unique state
        '''

        random = np.random.choice(self.moves)
        for S,x in enumerate(self.QTable[:,0]):
            if x == env:
                greedy = self.getGreedy(S)
                return (random,S) if np.random.random() < self.epsilon else (greedy,S)

        # Add env to table with array of randomised weights
        self.addState(env)
        return random, self.QTable.shape[0]-1     


    def updateQValues(self,SPrime, reward):
        '''
        Gets Q-values for state-action pairs and updates
        Q-table with the update equation

        Args: Sprime - latest state, i.e. S'
                reward - reward for last action
        '''
        # Q(s,a)
        Q = self.QTable[self.S][1][self.action]
        # Max Q(s',a') from legal moves
        QPrimeMax = self.QTable[SPrime][1][self.moves].max()

        # Increment N(s,a)
        self.QTable[self.S][2][self.action] += 1

        # Q learning update equation
        update = self.alpha*(reward + self.gamma*QPrimeMax - Q)
        self.QTable[self.S][1][self.action] += update

    def scores(self, state):
        '''
        Calculates reward given previous state from game score
        Reward is None for initial state, since no previous state

        Args: state - game state from Pac-Man game
        Returns: reward - score difference from previous state to current state
        '''
        newScore = state.getScore()
        reward = newScore - self.scoreLast
        self.scoreLast = newScore
        return reward

    def getAction(self, state):
        '''
        Main function to be called each time-step of the game.
        Gets legal moves and converts to numbers
        - Converts agent/food positions to a tuple representing the environment
        - Gets action based on environment
        - Updates Q-values of previous state/action pair with new reward
        - Store action and state for next time-step
        - Return selected action

        Args: state - game state from Pac-Man game
        Returns: actionPrime (as direction) - action number converted to Direction
        '''
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        self.moves = [self.dirToNum(move) for move in legal]

        # Get locations and create environment
        pacLoc = state.getPacmanPosition()
        gLoc = state.getGhostPositions()
        food = state.getFood()
        env = (pacLoc, gLoc, food)

        # Get action from Q-Table, add env if not seen
        actionPrime,SPrime = self.QAction(env) 

        # Update real scores from actions
        if self.action != None:
            reward = self.scores(state)
            self.updateQValues(SPrime,reward)

        # Log actions for next time-step
        self.action = actionPrime
        self.S = SPrime
        
        if actionPrime == None:
            raise Exception("No action selected")
        return self.numToDir(actionPrime)
            

    def final(self, state):
        '''
        Called at end of each win/loss.
        Updates Epsilon values to change exploration with each game
        '''

        # Reduce epsilon value each game end to reduce exporation
        self.setEpsilon(self.epsilon*0.995)

        # Update Q-Values from last step of game
        reward = self.scores(state)
        self.updateQValues(self.S,reward) 

        # Reset these variables
        self.moves = []
        self.scoreLast = None
        self.S = None
        self.action = None

        print "A game just ended!"
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)

