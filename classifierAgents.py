# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
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

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
from sklearn import tree
import math
import Queue

class Node(object):
    def __init__(self,value):
        self.value = value

    def predict(self,X):
        pass

class Root(Node):
    def __init__(self,value):
        self.value = value
        self.children = []
        self.attrIndex = None

    def predict(self,X):
        for child in self.children:
            if X[self.attrIndex] == child.value:
                return child.predict(X)

class Decision(Root):
    def __init__(self, value):
        super(Decision,self).__init__(value)

class Leaf(Node):
    def __init__(self,value,clss):
        self.clss = clss
        self.value = value

    def predict(self,X):
        return self.clss


class DecisionTree():  
    
    def __init__(self,minLeafInstances = 1):
        self.nodes = []
        self.minLeafInstances = minLeafInstances #min leaf samples
        self.classes = []

    def train(self,X,Y): 
        """ Trains decision tree

        Args:
            X (List): List of feature vectors/samples
            Y (List)): List of class labels

        Raises:
            ValueError: Number of labels should match samples
        """           
        X = np.array(X)
        Y = np.array(Y) 
        nSamples, self.nFeatures = X.shape
        if len(Y) != nSamples:
            raise ValueError("Number of labels {} does not match samples {}".format(len(Y),nSamples))

        yNew = self.trainRemoveConflicts(X,Y)
        self.nodes.append(Root("Root"))
        self.build(X, yNew, self.nodes[0])
    
    def trainRemoveConflicts(self,X,Y):
        """Converts instances with conflicting class labels to modal class

        Args:
            X (List): List of feature vectors
            Y (List): List of class labels

        Returns:
            Y (List): List of classes with conflicting classes as mode
        """        
        for row in X:
            matches = np.where((X == row).all(axis=1))[0]
            if len(matches) > 1:
                classes, counts = np.unique(Y[matches],return_counts=True)
                # make Y most common class
                Y[matches] = Y[np.argmax(counts)]   
        return Y

    def predict(self,X):
        """Predicts class from input feature vector from Pacman environment

        Args:
            X (List): Feature vector

        Returns:
            int : Predicted class of feature vector
        """        
        return self.nodes[0].predict(X)

    # instanceClasses are numbers of each class contained
    # simply works out purity of class
    def purity(self,instanceClasses):
        """Calculates purity of class

        Args:
            instanceClasses (List): Number of each class contained

        Returns:
            purity (float): Purity of the class
        """        
        M = float(sum(instanceClasses))
        purity = 0.0
        for n in instanceClasses:
            if n != 0:
                fraction = float(n)/M
                Bq = -(fraction*math.log(fraction,2))
            else:
                Bq = 0
            purity += Bq

        return purity


    def weightedEntropy(self,featureColumn,classColumn):
        """Calculates the weighted entropy of a feature and its class

        Args:
            featureColumn (List): column of features
            classColumn (List): Class labels of the features

        Returns:
            remainder (float): Weighted entropy of data
        """
        
        features = list(np.unique(featureColumn)) # make list of unique features
        
        featCounter = np.zeros(len(features)) # counter for each features
        yesCounter = featCounter.copy() # counter for feature and yes

        for f,c in zip(featureColumn,classColumn):
            featCounter[features.index(f)] += 1
            if c != 0 or c != False:
                yesCounter[features.index(f)] += 1 

        M = len(featureColumn)
        remainder = 0
        for feat,yes in zip(featCounter,yesCounter):
            remainder += feat/M * self.purity([yes,feat-yes])
        return remainder    


    def allSame(self,arry):
        """Checks if all elements in an array are equal

        Args:
            arry (numpy array): Array to be checked

        Returns:
            (bool): True if equal, False if not equal
        """        
        if len(arry)==1:
            return True
        else:
            return all(np.array_equal(el,arry[0]) for el in arry)

    def build(self, X, Y, parentNode):
        """Recursive splitting of dataset in training
         to create Leaf or Decision Nodes.
         This builds the tree in training.

        Args:
            X (List): List of feature vectors/instance rows
            Y (List): List of labels
            parentNode (Leaf or Decision class): Pointer to parent node of current dataset
        """        

        reshapeX = np.array(X).T 

        attrEntropies = []
        for col in reshapeX: 
            # calc entropies
            attrEntropies.append(self.weightedEntropy(col,Y)) 

        # max information gain is min entropy
        maxIndex = np.argmin(attrEntropies)
        parentNode.attrIndex = maxIndex

        # unique attributes we will split by
        newSets = np.unique(reshapeX[maxIndex]) 

        # check if entropies are the same
        entsIsSame = self.allSame(attrEntropies)
        for att in newSets:
            # xNew,yNew are values with split attribute
            rows = np.where(reshapeX[maxIndex]==att) 
            xNew = X[rows] 
            yNew = Y[rows]

            if self.allSame(yNew) or entsIsSame:
                # if all classes, features or entropies are the same make a leaf node
                leafNode = Leaf(att,yNew[0])
                parentNode.children.append(leafNode)
                self.nodes.append(leafNode)
            else:
                # make a decision node
                decNode = Decision(att) 
                parentNode.children.append(decNode)
                self.nodes.append(decNode)
                self.build(xNew,yNew,decNode)





# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"
        self.model = DecisionTree()
    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray
                
    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of imtegers 0-3 indicating the action
        # taken in that state.
            
        # *********************************************
        #
        # Any other code you want to run on startup goes here.
        #
        # You may wish to create your classifier here.
        #
        # *********************************************

        # Train the model
        self.model.train(self.data,self.target)
        
    # Tidy up when Pacman dies
    def final(self, state):

        print "I'm done!"
        
        # *********************************************
        #
        # Any code you want to run at the end goes here.
        #
        # *********************************************

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state) # first 4: walls, next 4: food, next 8: ghosts, next 1: ghost infront, next 1: class
        
        # *****************************************************
        #
        # Here you should insert code to call the classifier to
        # decide what to do based on features and use it to decide
        # what action to take.
        #
        # *******************************************************

        # from collected 'features' vector, classify as any of 0-3.
        # this gives your move
        
        # Get class from model and convert to action
        number = self.model.predict(features)
        action = self.convertNumberToMove(number)

        # Get the actions we can try.
        legal = api.legalActions(state)

        return api.makeMove(action,legal)
        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are. We need to pass
        # the set of legal moves to teh API so it can do some safety
        # checking.
        # return api.makeMove(Directions.STOP, legal)

