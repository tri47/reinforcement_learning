import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import random as rand  		   	  		
import time
import math

# Class for Q-learner
# alpha: learning rate
# gamma: discount rate
# exr: exploration rate 
# excr: exploration cooling rate
# partially apdated from ML4T course, Geogia Tech
class QLearner(object):
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2,\
        gamma = 0.9, \
        exr = 0.5, \
        excr = 0.99 ):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
        self.num_actions = num_actions  		
        self.num_states = num_states   	  			  	 		  		  		    	 		 		   		 		  
        self.s = 0  		   	  			  	 		  		  		    	 		 		   		 		  
        self.a = 0  	
        self.q = np.zeros([num_states,num_actions])	 # init Q table
        self.alpha = alpha
        self.gamma = gamma
        self.exr = exr
        self.excr = excr
  		   	  			  	 		  		  		    	 		 		   		 		  
    def querysetstate(self, s):  		   	  			  	 		  		  		    	 		 		   		 		  		   	  			  	 		  		  		    	 		 		   		 		  
        self.s = s  
        action_Qs = self.q[s]
        action = np.argmax(action_Qs) # pick best action
        # choose random action based on exploration rate or if there is no best action
        if (rand.uniform(0.0, 1.0) <= self.exr) or (action_Qs.sum() == 0):
            action = rand.randint(0, self.num_actions-1)  	
        self.a = action	   	  			  	 		  	
        self.exr = self.exr*self.excr 		  		    	 		 		   		 		  
        return action  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def query(self,s_prime,r):  		   	  			  	 		  		  		    	 		 		   		 		  	   	  
        alpha = self.alpha
        gamma = self.gamma
        s = self.s
        a = self.a
        action_Qs = self.q[s_prime] # Q values for available actions in new state       
        max_future_reward = action_Qs.max()
        future_reward = alpha*(r + gamma*max_future_reward)
        self.q[s,a] = (1-alpha)*self.q[s,a] + future_reward
        action = self.querysetstate(s_prime)		   	
        return action