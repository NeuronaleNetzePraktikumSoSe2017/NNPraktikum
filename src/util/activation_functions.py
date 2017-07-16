# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

from numpy import exp
from numpy import divide
from numpy import sum
from numpy import max


class Activation:
    """
    Containing various activation functions and their derivatives
    """

    @staticmethod
    def sign(netOutput, threshold=0):
        return netOutput >= threshold

    @staticmethod
    def sigmoid(netOutput):
        # Here you have to code the sigmoid function
        return divide(1.0, (1.0 + exp(-netOutput)))

    @staticmethod
    def sigmoidPrime(netOutput):
        # Here you have to code the derivative of sigmoid function
        # netOutput.*(1-netOutput)
        return Activation.sigmoid(netOutput) * (1.0 - Activation.sigmoid(netOutput))

    @staticmethod
    def tanh(netOutput):
        # Here you have to code the tanh function
        return divide(1.0 - exp(-2.0 * netOutput), 1.0 + exp(-2.0 * netOutput))

    @staticmethod
    def tanhPrime(netOutput):
        # Here you have to code the derivative of tanh function
        return 1.0 - Activation.tanh(netOutput)^2 #TODO verifizieren.

    @staticmethod
    def rectified(netOutput):
        return lambda x: max(0.0, x)

    @staticmethod
    def rectifiedPrime(netOutput):
        # Here you have to code the derivative of rectified linear function
        if netOutput <= 0: #TODO: interpretation at 0. It might be undefined.
            return 0
        else:
            return 1

    @staticmethod
    def identity(netOutput):
        return lambda x: x

    @staticmethod
    def identityPrime(netOutput):
        # Here you have to code the derivative of identity function
        return 1

    @staticmethod
    def softmax(netOutput):
        # Here you have to code the softmax function
        #numerator = [exp(i) for i in netOutput] #TODO: is the expected return value an array or a number?
        #sum_numerator = sum(numerator)
        #softmax = [i / sum_numerator for i in netOutput]
        #return softmax

        result_unnormalized = exp(netOutput - max(netOutput))
        normalization = sum(result_unnormalized)
        return divide(result_unnormalized, normalization)


    @staticmethod
    def getActivation(str):
        """
        Returns the activation function corresponding to the given string
        """

        if str == 'sigmoid':
            return Activation.sigmoid
        elif str == 'softmax':
            return Activation.softmax
        elif str == 'tanh':
            return Activation.tanh
        elif str == 'relu':
            return Activation.rectified
        elif str == 'linear':
            return Activation.identity
        else:
            raise ValueError('Unknown activation function: ' + str)

    @staticmethod
    def getDerivative(str):
        """
        Returns the derivative function corresponding to a given string which
        specify the activation function
        """

        if str == 'sigmoid':
            return Activation.sigmoidPrime
        elif str == 'tanh':
            return Activation.tanhPrime
        elif str == 'relu':
            return Activation.rectifiedPrime
        elif str == 'linear':
            return Activation.identityPrime
        else:
            raise ValueError('Cannot get the derivative of'
                             ' the activation function: ' + str)
