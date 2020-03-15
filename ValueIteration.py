import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import random as rand  		   	  		
import time
import math

# Class for Value Iteration
# gamma: discount rate
class ValueIteration(object):
    def __init__(self, gamma=0.9,\
        num_states=100, \
        num_actions=4, \
        R_f = None, \
        return_max_V = None, env = None, size=20 ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.V_star = np.zeros([num_states, 1])
        self.Pi_star = np.zeros([num_states, 1])
        self.R_f = R_f
        self.return_max_V = return_max_V
        self.env = env
        self.size= size

    def iterate_V(self):
        new_V = self.V_star.copy()
        for i in range(0,new_V.shape[0]):
            new_V[i][0] = self.R_f[i][0] + self.gamma*self.return_max_V(self.env, self.V_star, i,size=self.size) 
        self.V_star = new_V
        return self.V_star

    def return_policy(self):
        for i in range(0,self.Pi_star.shape[0]):
            self.Pi_star[i][0] = self.return_max_V(self.env, self.V_star, i, mode ='p',size=self.size) 
        return self.Pi_star