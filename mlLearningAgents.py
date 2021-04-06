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
    def __init__(self, alpha=0.01, epsilon=0.05, gamma=0.8, numTraining = 1000):
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

        self.table = np.empty((0,3))
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
        
        pass

   
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def setAlpha(self, value):
        self.alpha = value
        

    def addState(self,env):
        numActions = 4
        #[state,Q value, N(s,a)] randomised Q values
        # row = np.array([[env,np.random.rand(numActions),np.zeros(numActions)]])
        row = np.array([[env,np.zeros(numActions),np.zeros(numActions)]])
        self.table = np.append(self.table,row, axis=0)
        
    def getGreedy(self,S):
        Qs = self.table[S][1]
        Ns = self.table[S][2]
        product = Qs*Ns
        sort = reversed(np.argsort(product))
        for x in sort:
            if x in self.moves: return x
        raise Exception("Chosen action not legal")

    def QAction(self,env):
        # returns action to take given Qtable
        # random action for exploration
        random = np.random.choice(self.moves)
        # check for state in table's Env column
        S = None
        for index,x in enumerate(self.table[:,0]):
            if x == env:
                S = index
                greedy = self.getGreedy(S)
                return (random,S) if np.random.random() < self.epsilon else (greedy,S)

        # add env to table with array of randomised weights
        self.addState(env)
        # since new state, return random action and last env
        return random, self.table.shape[0]-1     


    def NSA(self,action):
        self.table[self.S][2][action] += 1 
        return self.table[self.S][2][action]

    def updateWeights(self,SPrime, reward):
        q = self.table[self.S][1][self.action]
        qPrimeMax = self.table[SPrime][1][self.moves].max()

        #increment N(s,a)
        Nsa = self.NSA(self.action)

        update = self.alpha*Nsa*(reward + self.gamma*qPrimeMax - q)
        self.table[self.S][1][self.action] += update

    def numToDirection(self,num):
        return self.dirDic[num]

    def dirToNum(self,direction):
        return self.dirDic.values().index(direction)

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        self.moves = [self.dirToNum(move) for move in legal]

        # get states
        pacLoc = state.getPacmanPosition()
        gLoc = state.getGhostPositions()
        food = state.getFood()
        env = (pacLoc, gLoc, food)


        # LOOKUP ACTION
        # add state if not seen, and init N(s,a)'s to 0
        actionPrime,SPrime = self.QAction(env) # action(a') is integer 0-3, SPrime is index of state

        # Update real scores from actions
        newScore = state.getScore()
        if self.scoreLast != None:
            reward = newScore-self.scoreLast
            self.updateWeights(SPrime,reward)

        self.scoreLast = newScore
        self.action = actionPrime
        self.S = SPrime
        
        # print "Legal moves: ", legal
        # print "Pacman position: ", state.getPacmanPosition()
        # print "Ghost positions:" , state.getGhostPositions()
        # print "Food locations: "
        # print state.getFood()
        # print "Score: ", state.getScore()
            
        # Now pick what action to take. For now a random choice among
        # the legal moves
        if actionPrime == None:
            actionPrime = random.choice(legal)
        # We have to return an action
        return self.numToDirection(actionPrime)
            

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):
        
        self.setEpsilon(self.epsilon*0.99)

        reward = state.getScore() - self.scoreLast
        self.updateWeights(self.S,reward) 

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

            print(self.epsilon)

            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)

