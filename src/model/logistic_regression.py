# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from model.logistic_layer import LogisticLayer

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        # self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

        self.layer = LogisticLayer(self.trainingSet.input.shape[1], 1, activation='sigmoid')

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        from util.loss_functions import BinaryCrossEntropyError
        loss = BinaryCrossEntropyError()

        nextWeights = np.ones((1,1))

        for iteration in range(self.epochs):
            grad = 0
            index = np.random.randint(len(self.trainingSet.input))
            input = self.trainingSet.input[index]
            label = self.trainingSet.label[index]
            predictedLabel = np.clip(self.fire(input), 1e-8, 1 - 1e-8)
            error = loss.calculateError(label, predictedLabel)
            # derivative of binary cross entropy
            lossDerivative = np.divide((predictedLabel - label), predictedLabel * (1.0 - predictedLabel))
            self.layer.computeDerivative(lossDerivative * predictedLabel, nextWeights)
            self.updateWeights()
            
            if verbose:
                logging.info("Epoch: %i; Error: %f", iteration, error)


    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.fire(testInstance) > 0.5

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self):
        self.layer.updateWeights(self.learningRate)

    def fire(self, input):
        inputArray = np.ndarray((1, self.layer.nIn + 1))
        inputArray[0][0] = 1
        inputArray[0][1:] = input
        return self.layer.forward(inputArray)[0][0]
