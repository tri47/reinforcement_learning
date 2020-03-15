import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import random as rand  		   	  		
import time
import math

# Class for Policy Iteration
# gamma: discount rate
class PolicyIteration(object):
    def __init__(self, gamma=0.9,\
        num_states=100, \
        num_actions=4, \
        R_f = None, \
        return_max_V = None, env = None, size =20 ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.V_star = np.zeros([num_states, 1])
        self.Pi_star = np.zeros([num_states, 1])
        self.R_f = R_f
        self.return_max_V = return_max_V
        self.env = env
        self.size = size

    def iterate_P(self):
        new_V = self.V_star.copy()
        diff = -1
        while diff != 0:
            old_V = new_V.copy()
            for i in range(0,new_V.shape[0]):
                new_V[i][0] = self.R_f[i][0] + self.gamma*self.return_max_V(self.env, self.V_star, i, a_in = self.Pi_star[i][0], size=self.size) 
            self.V_star = new_V
            diff = np.absolute((self.V_star - old_V)).max()
           # print(new_V)
          #  print(old_V)
        self.V_star = new_V

        for i in range(0,self.Pi_star.shape[0]):
            self.Pi_star[i][0] = self.return_max_V(self.env, self.V_star, i, mode ='p', size=self.size) 
        return self.Pi_star