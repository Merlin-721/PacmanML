import unittest
from classifierAgents import DecisionTree
import numpy as np

class test_DecisionTree(unittest.TestCase):

    def test_get_purity(self):
        self.tree = DecisionTree()
        input =[6, 4]
        test = self.tree.purity(input)
        answer = 0.97
        self.assertEqual(answer,round(test,2))

    def test_information_gain(self):
        self.tree = DecisionTree()
        Y = [True,True,False,True,True,True,False,False,True,False]
        X = ["Action","Comedy","Drama","Comedy","Action","Drama","Comedy","Action","Drama","Action"]
        answer = 0.951

        test = self.tree.informationGain(X,Y)
        self.assertEqual(answer,np.round(test,3))

    