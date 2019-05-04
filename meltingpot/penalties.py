import numpy as np

class Penalty:
    """ class for penalty function """
    __slots__ = ['alpha','beta','C']

    def __init__(self,alpha=2,beta=2,C=100):
        self.alpha = alpha
        self.beta = beta
        self.C = C

    def eval_ics(self,values,ii):
        """ evaluate penalty term for inequality constraints"""

        p = np.zeros(values.shape,dtype=float)
        idx = np.nonzero(values>0)
        p[idx] = ((ii+1)*self.C)**(self.alpha) * values[idx]**self.beta

        return p

    def eval_ecs(self,values,ii):
        """ evaluate penalty term for equality constraints"""

        p = np.zeros(values.shape,dtype=float)
        idx = np.nonzero(values)
        p[idx] = ((ii+1)*self.C)**(self.alpha) * np.abs(values[idx])**self.beta

        return p
