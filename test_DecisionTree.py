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

        test = self.tree.weightedEntropies(X,Y)
        self.assertEqual(answer,np.round(test,3))

    def test_split(self):

        testX = {}
        testY = {}
        new = np.array([False, False, True, True, False, False, False, True, True, True])
        type = np.array(["Action", "Comedy", "Drama", "Comedy", "Action", "Drama", "Comedy", "Action", "Drama", "Action"])
        lang = np.array(["Eng", "Sp", "Eng", "Sp", "Sp", "Sp", "Fr", "Sp", "Eng", "Fr"])
        X = np.array([new, type, lang]).transpose()

        # class labels
        Y = np.array([True, True, False, True, True, True, False, False, True, False])

        # MANUALLY SPLIT SETS - split by Language

        # eng split
        newEng = [False, True, True]
        typeEng = ["Action", "Drama", "Drama"]
        xEng = np.array([newEng, typeEng]).transpose()
        yEng = np.array([True, False, True])

        testX["Eng"] = xEng
        testY["Eng"] = yEng

        # sp split
        newSp = [False, True, False, False, True]
        typeSp = ["Comedy",  "Comedy", "Action", "Drama", "Action"]
        xSp = np.array([newSp, typeSp]).transpose()
        ySp = [True, True, True, True, False]

        testX["Sp"] = xSp
        testY["Sp"] = ySp

        newFr = [False,  True]
        typeFr = ["Comedy", "Action"]
        xFr = np.array([newFr, typeFr]).transpose()
        yFr = [False, False]

        testX["Fr"] = xFr
        testY["Fr"] = yFr

        test = {}
        test["X"] = testX
        test["Y"] = testY

        # class labels
        self.tree = DecisionTree()
        result = self.tree.split(X, Y)
        self.assertItemsEqual(result, test)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_DecisionTree("test_split"))
    return suite
if __name__ == '__main__':
    result = suite()
    result.debug()




#     {'y': {
#         'Fr': [False, False], 
#         'Sp': [True, True, True, True, False], 
#         'Eng': array([ True, False,  True])}, 

#      'x': {
#          'Fr': array([['False', 'Comedy'], ['True', 'Action']], dtype='|S6'),
#          'Sp': array([['False', 'Comedy'], ['True', 'Comedy'], ['False', 'Action'], ['False', 'Drama'], ['True', 'Action']], dtype='|S6'), 
#          'Eng': array([['False','Action'], ['True', 'Drama'],['True
