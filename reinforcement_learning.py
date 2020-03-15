import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import random as rand  	
import matplotlib.pyplot as plt	   	  			  	 		  		  		    	 		 		   		 		  
import time  		   	  			  	 		  		  		    	 		 		   		 		  
import math  		   	  			  	 		  		  		    	 		 		   		 		  
import Qlearner as ql  		 
import ValueIteration as vl  	
import PolicyIteration as pl
from matplotlib.ticker import FormatStrFormatter

# maze problem partially adapted from Machine learning for Trading course, Georgia Tech
# print out the map  		   	  			  	 		  		  		    	 		 		   		 		  
def printmap(data):  		   	  			  	 		  		  		    	 		 		   		 		  
    print("--------------------")  		   	  			  	 		  		  		    	 		 		   		 		  
    for row in range(0, data.shape[0]):  		   	  			  	 		  		  		    	 		 		   		 		  
        for col in range(0, data.shape[1]):  		   	  			  	 		  		  		    	 		 		   		 		  
            if data[row,col] == 0: # Empty space  		   	  			  	 		  		  		    	 		 		   		 		  
                print(".", end=' ')  		   	  			  	 		  		  		    	 		 		   		 		  
            if data[row,col] == 1: # Columns  		   	  			  	 		  		  		    	 		 		   		 		  
                print("O", end=' ')  		   	  			  	 		  		  		    	 		 		   		 		  
            if data[row,col] == 2: # Starting position 		   	  			  	 		  		  		    	 		 		   		 		  
                print("S", end=' ')  		   	  			  	 		  		  		    	 		 		   		 		  
            if data[row,col] == 3: # Goal  		   	  			  	 		  		  		    	 		 		   		 		  
                print("G", end=' ')  		   	  			  	 		  		  		    	 		 		   		 		  
            if data[row,col] == 4: # Trail  		   	  			  	 		  		  		    	 		 		   		 		  
                print("x", end=' ')  		   	  			  	 		  		  		    	 		 		   		 		  
            if data[row,col] == 5: # Quick sand  		   	  			  	 		  		  		    	 		 		   		 		  
                print("P", end=' ')  		   	  			  	 		  		  		    	 		 		   		 		  
            if data[row,col] == 6: # Stepped in quicksand  		   	  			  	 		  		  		    	 		 		   		 		  
                print("@", end=' ')  		   	  			  	 		  		  		    	 		 		   		 		  
        print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print("--------------------")  	

def printpolicy(data,policy):  		   	  			  	 		  		  		    	 		 		   		 		  
    print("--------------------")  		   	  			  	 		  		  		    	 		 		   		 		  
    for row in range(0, policy.shape[0]):  		   	  			  	 		  		  		    	 		 		   		 		  
        for col in range(0, policy.shape[1]):  	
            if data[row,col] == 1:
                print("O", end = ' ')
            elif data[row,col] == 5:
                print("P", end=' ') 
        #    elif data[row,col] == 2:
         #       print("S", end=' ') 
            elif data[row,col] == 3:
                print("G", end=' ') 
            else:		  	 		  		  		    	 		 		   		 		  
                if policy[row,col] == 0: # Empty space  		   	  			  	 		  		  		    	 		 		   		 		  
                    print("^", end=' ')  		   	  			  	 		  		  		    	 		 		   		 		  
                if policy[row,col] == 1: # Obstacle  		   	  			  	 		  		  		    	 		 		   		 		  
                    print(">", end=' ')  		   	  			  	 		  		  		    	 		 		   		 		  
                if policy[row,col] == 2: # Starting position 		   	  			  	 		  		  		    	 		 		   		 		  
                    print("V", end=' ')  		   	  			  	 		  		  		    	 		 		   		 		  
                if policy[row,col] == 3: # Goal  		   	  			  	 		  		  		    	 		 		   		 		  
                    print("<", end=' ')  		   	  			  	 		  		  		    	 		 		   		 		  		   	  			  	 		  		  		    	 		 		   		 		  
        print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print("--------------------")  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
# find current position of the robot		   	  			  	 		  		  		    	 		 		   		 		  
def getrobotpos(data):  	
    results = np.where(data==2)	  
    # Return the the indices of the position
    return [*zip(results[0],results[1])][0] 	  			  	 		  		  		    	 		 		   		 		  		   	  			  	 		  		  		    	 		 		   		 		     	  			  	 		  		  		    	 		 		   		 		  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
# find position of the goal		   	  			  	 		  		  		    	 		 		   		 		  
def getgoalpos(data):  		   	  			  	 		  		  		    	 		 		   		 		    	  			  	 		  		  		    	 		 		   		 		  
    results = np.where(data==3)	  
    # Return the the indices of the position
    return [*zip(results[0],results[1])][0] 	   	  			  	 		  		  		    	 		 		   		 		  

def movebot(maze,lastpos,a):
    row, col = lastpos
    reward = -1 # default reward
    penalty = -1000
    goal_reward = 1000
    fail_rate = 0.1 # prob. that the intended action is unsuccessful
    if rand.uniform(0,1) <= fail_rate:
        if a == 0: a = rand.choice([1,3])
        elif a == 1: a = rand.choice([0,2])
        elif a == 2: a = rand.choice([1,3])
        elif a == 3: a = rand.choice([0,2])
        #a = rand.randint(0,3)
    if a == 0: row -= 1 # Moving north
    elif a == 1:  col += 1 # Moving east
    elif a == 2:  row += 1 # Moving south
    elif a == 3:  col -= 1 # Moving west
    # dealing with edge cases: off the maze
    if (row < 0) or (col < 0) or (row >= maze.shape[0]) or (col >= maze.shape[1]):
        row, col = lastpos
    elif maze[row,col] == 1: # columns
        row, col = lastpos
    elif maze[row,col] == 5: # pits
        reward = penalty
    elif maze[row, col] == 3:
        reward = goal_reward
    return (row, col), reward
    	  		  		    	 		 		   		 		  
# Represent the position as an integer state		   	  			  	 		  		  		    	 		 		   		 		  
def discretize(pos,size):  		   	  			  	 		  		  		    	 		 		   		 		  
    return pos[0]*size + pos[1]  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def train_q_learner(maze, episodes, learner, verbose, size=10, run_time = 10000, random_start = False):  		   	  			  	 		  		  		    	 		 		   		 		  		   	  			  	 		  		  		    	 		 		   		 		  
    startpos = getrobotpos(maze)  
    goalpos = getgoalpos(maze) 		   	  			  	 		  		  		    	 		 		   		 		  
    scores = np.zeros((episodes,1))  
    diffs = np.zeros((episodes,1))		   	  			  	 		  		  		    	 		 		   		 		  
    for episode in range(0,episodes):  		   	  			  	 		  		  		    	 		 		   		 		  
        total_reward = 0  	
        if random_start:
            start_row = rand.randint(0,size-1)
            start_col = rand.randint(0,size-1)
            while (maze[start_row,start_col] != 0):
                start_row = rand.randint(0,size-1)
                start_col = rand.randint(0,size-1)  
            startpos = (start_row,start_col)
        
        data = maze.copy()  		   	  			  	 		  		  		    	 		 		   		 		  
        robopos = startpos  		   	  			 	 		  		  		    	 		 		   		 		  
        state = discretize(robopos,size) #convert the location to a state  		   	  			  	 		  		  		    	 		 		   		 		  
        action = learner.querysetstate(state) #set the state and get first action  		   	  			  	 		  		  		    	 		 		   		 		  
        old_q = learner.q.copy()
        count = 0
        while count<run_time:
            #move to new location according to action and then get a new action  		   	  			  	 		  		  		    	 		 		   		 		  
            newpos, r = movebot(data,robopos,action)  
            state = discretize(newpos,size)  		   	  			  	 		  		  		    	 		 		   		 		  
            action = learner.query(state,r)	  	 		  		  		    	 		 		   		 		  
            if data[newpos] != 5 and data[newpos] != 2 and data[newpos] != 3:  		   	  			  	 		  		  		    	 		 		   		 		  
                data[newpos] = 4 # mark trail	  	 		  		  		    	 		 		   		 		  
            robopos = newpos # update the location  		   	  			  	 		  		  		    	 		 		   		 		  
            total_reward += r 	  			  	 		  		  		    	 		 		   		 		  
            count = count + 1  		 
            if data[robopos] == 5:
                data[robopos] = 6
                break  	  	
            if data[robopos] == 3:
                break	
        new_q = learner.q.copy()
        
        diff = np.absolute((new_q- old_q)).max()   
        diffs[episode,0] = diff 	   	  			  	 		  		  		    	 		 		   		 		  
    return diffs   	  			  	 		  		  		    	 		 		   		 		  	   	  			  	 		  		  		    	 		 		   		 		  
 
def return_max_V_maze(maze, V_star, s, mode = 'v', a_in = None,size=10):
    lastrow = s//size 
    lastcol = s%size 
    fail_rate = 0.10
    max_v = None
    max_a = None
    avai_actions = [0,1,2,3]
    if a_in is not None:
        avai_actions = [a_in]
    for main_a in avai_actions:
        max_Vs = []
        if (main_a == 0) or (main_a == 2): actions = [main_a,1,3]
        elif (main_a == 1) or (main_a == 3): actions = [main_a,0,2]

        for a in actions:
            row = lastrow
            col = lastcol
            if a == 0: row = lastrow - 1 # Moving north
            elif a == 1:  col = lastcol + 1 # Moving east
            elif a == 2:  row = lastrow + 1 # Moving south
            elif a == 3:  col = lastcol - 1 # Moving west
            new_s = discretize((row,col),size)
            # dealing with edge cases: off the maze
            if (row < 0) or (col < 0) or (row >= maze.shape[0]) or (col >= maze.shape[1]):
                new_s = s 
            elif maze[row,col] == 1:
                new_s = s
            max_V = V_star[new_s][0]
            max_Vs.append(max_V)        
        v = (1-fail_rate)*max_Vs[0] + fail_rate/2*(max_Vs[1] + max_Vs[2])
        if (maze[lastrow,lastcol] == 3) or (maze[lastrow,lastcol] == 5):
            v = 0
        if max_v is None: max_v = v
        if max_a is None: max_a = main_a
        if v > max_v:
            max_v = v
            max_a = main_a
    if mode == 'v':
        return max_v
    elif mode == 'p':
        return max_a


def run_maze_problem(map_name, size, random_start = False):
    q_diffs_explore = np.array((1,1))
    q_diffs_exploit = np.array((1,1))
    diffs_value = np.array((1,1))
    diffs_policy = np.array((1,1))

    verbose = True 	
    penalty = -1000	   	  			
    goal_reward = 1000  	 		  		  		    	 		 		   		 		    			  	 		  		  		    	 		 		   		 		  
    # read map	   	  			  	 		  		  		    	 		 		   		 		  
    inf = open(map_name)  		   	  			  	 		  		  		    	 		 		   		 		  
    data = np.array([[*map(int,s.strip().split(','))] for s in inf.readlines()])  		   	  			  	 		  		  		    	 		 		   		 		  
    originalmap = data.copy() #make a copy so we can revert to the original map later  		   	  			  	 		  		  		    	 		 		   		 		     	  			  	 		  		  		    	 		 		   		 		  
    print('This is the map of the maze!')
    printmap(data)  		   	  	
    print('')		  	 		  		  		    	 		 		   		 		     	  			  	 		  		  		    	 		 		   		 		  
    rand.seed(5)

    # Make the reward matrix
    R_f = np.zeros([size**2,1])
    for row in range(0,data.shape[0]):
        for col in range(0,data.shape[1]):
            pos = (row, col)
            val = data[pos]
            reward = -1
            if val == 5: reward = penalty
            elif val == 3: reward = goal_reward
            s = discretize(pos,size)      
            R_f[s][0] = reward

    
    # Q LEARNER EXPLORE
    learner = ql.QLearner(num_states=size**2, num_actions = 4, \
        alpha = 0.2, gamma = 0.9, exr = 0.9, excr = 0.998) 		   	  			  	 		  		  		    	 		 		   		 		  
    episodes = 1200
    if random_start: episodes = 20000	  			  	 		  		  		    	 		 		   		 		  
    q_diffs_explore = train_q_learner(data, episodes, learner, verbose, size=size, run_time = 10000, random_start = random_start)  		   	  			  	 		  		  		    	 		 		   		 		   
  #  print(learner.q)
    Pi_star = np.zeros([size**2, 1])
    for i in range(0, Pi_star.shape[0]):
        Pi_star[i][0] = np.argmax(learner.q[i])  	
    print('This is the policy given by Q Learner Explorer!')		
    printpolicy(data,(Pi_star.reshape(size,size)))
    print('')

    
    # Q LEARNER EXPLOIT
    learner = ql.QLearner(num_states=size**2, num_actions = 4, \
        alpha = 0.2, gamma = 0.9, exr = 0.6, excr = 0.99) 		   	  			  	 		  		  		    	 		 		   		 		  
    episodes = 1500	  			  	 		  		  		    	 		 		   		 		  
    q_diffs_exploit = train_q_learner(data, episodes, learner, verbose, size=size, run_time = 10000) 
    Pi_star = np.zeros([size**2, 1])   
    for i in range(0, Pi_star.shape[0]):
        Pi_star[i][0] = np.argmax(learner.q[i])  	
    print('This is the policy given by Q Learner Exploiter!')		
    printpolicy(data,np.round_(Pi_star.reshape(size,size)))
    print('')
    
    
    # VALUE LEANER
    learner = vl.ValueIteration(gamma=0.9, num_states=size**2, num_actions=4, R_f = R_f, \
         return_max_V = return_max_V_maze, env = data, size=size)
    episodes = 300
    diffs_value = np.zeros((episodes,1))
    #for episode in range(0,episodes):
    diff = -1
    episode = 0
    while diff != 0:
        V_old = learner.V_star.copy()
        learner.iterate_V()
        V_new = learner.V_star.copy()
        diff = np.absolute((V_new - V_old)).max()
        diffs_value[episode,0] = diff
        episode += 1
      #  print(episode)
    policy = learner.return_policy()


    # printpolicy(data, policy.reshape(size,size))
    print('This is the policy given by Value Iterationer!')		
    printpolicy(data,np.round_(policy.reshape(size,size)))
    print('')

    # POLICY LEARNER
    learner = pl.PolicyIteration(gamma=0.9, num_states=size**2, num_actions=4, R_f = R_f, \
         return_max_V = return_max_V_maze, env = data,size=size)
    episodes = 20
    diffs_policy = np.zeros((episodes,1))
    for episode in range(0,episodes):
        print(episode)
        P_old = learner.Pi_star.copy()
        learner.iterate_P()
        P_new = learner.Pi_star.copy()
        diff = np.absolute((P_new - P_old)).max()
        diffs_policy[episode,0] = diff   
    print('This is the policy given by Policy Iterationer!')	         
    printpolicy(data,(learner.Pi_star.reshape(size,size)))
    print('')
    
    
    # PLOT RESULTS
    fig, axes = plt.subplots(2,2)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.35)

    ax = axes[1,0]
    ax.plot(q_diffs_explore)
    ax.title.set_text('Exploring Q Learner (explore rate = 0.9)')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max abs. update on Q table')

    ax = axes[1,1]
    ax.plot(q_diffs_exploit)
    ax.title.set_text('Exploiting Q Learner (explore rate = 0.6)')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max abs. update on Q table')

    ax = axes[0,0]
    ax.plot(diffs_value)
    ax.title.set_text('Value Iteration')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xlabel('Iteration')
    ax.set_yscale('log')
    ax.set_ylabel('Max abs. update on Value table')

    ax = axes[0,1]
    ax.plot(diffs_policy)
    ax.title.set_text('Policy Iteration')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max abs. update on Policy table')

    plt.show()
    


# PROBLEM 2
def printsequence(sequence):
    print('___________________________')
    for i in range(0,sequence.shape[0]):
        if sequence[i][0] == 0:
            print('L', end=' ')
        else:
            print('R', end = ' ')
    print('\n___________________________')

def turn_dial(sequence, s, a):
    minor_reward = 2
    major_reward = 100
    sequence_cs = np.cumsum(sequence)
    stages = sequence.shape[0]
    good_action = 0
    reward = 0
    next_s = s
    for i in range(0,stages):
        if s >= sequence_cs[i]: 
            good_action = 1 - good_action
            continue
        else:
            if a != good_action:
                reward = minor_reward
                next_s = -1 # mark minor reward
            else:
                reward = 0
                next_s = s+1
    if next_s == sequence_cs[-1] : 
        reward = 100
    return next_s, reward    

def train_lock_learner(sequence, episodes, learner):	   	  			  	 		  		  		    	 		 		   		 		  
    scores = np.zeros((episodes,1))  
    diffs = np.zeros((episodes,1))
    for episode in range(0,episodes):  		   	  			  	 		  		  		    	 		 		   		 		  
        state = 0
        last_state = sum(sequence)
        action = learner.querysetstate(state) 
        old_q = learner.q.copy()
        count = 0
        while count<10000:
            #move to new location according to action and then get a new action  	   	  			  	 		  		  		    	 		 		   		 		  
            new_s, r = turn_dial(sequence,state,action)  
           # print('new state',new_s) 	
            action = learner.query(new_s,r)    
            if new_s == -1 or new_s == last_state:	   	  	
                break		  	 		  		  		    	 		 		   		 		   
          #  print('action ', action)		 		   		 		  
            state = new_s # update state	 		  		  		    	 		 		   		 		  
            count = count + 1  
        new_q = learner.q.copy()
        diff = np.absolute(new_q - old_q).max()
        diffs[episode,0] = diff
    return diffs
    
def return_max_V_lock(sequence, V_star, s, mode = 'v', a_in = None,size=10):
    major_reward = 100
    minor_reward = 2
    last_state = sum(sequence) + 1 # minor reward state
    sequence_cs = np.cumsum(sequence)
    stages = sequence.shape[0]
    max_v = 0
    max_a = 0
    if (s == last_state) or (s == (last_state -1)): 
        max_v = 0
    else:
        next_s = s + 1
        max_v = max([V_star[next_s][0], V_star[-1][0]]) #V_star[next_s][0]
        good_action = 0
        for i in range(0,stages):
            if s >= sequence_cs[i]: 
                good_action = 1 - good_action
        if good_action == 0: 
            max_a = np.argmax([V_star[next_s][0], V_star[-1][0]])
        if good_action == 1: max_a = np.argmax([V_star[-1][0], V_star[next_s][0]])
        if a_in is not None:
            if a_in != good_action:
                max_v = minor_reward
            else:
                max_v = V_star[next_s][0]
    if mode == 'v':
        return max_v
    elif mode == 'p':
        return max_a


def run_lock_problem():
    q_diffs_explore = np.array((1,1))
    q_diffs_exploit = np.array((1,1))
    diffs_value = np.array((1,1))
    diffs_policy = np.array((1,1))

    sequence = np.array([1,2,1,2])
    minor_reward = 2
    major_reward = 100
    state_count = sum(sequence) + 1
    
    # Q EXPLORE # FIND CORRECT SEQUENCE 90% OF TIME
    learner = ql.QLearner(num_states=state_count, num_actions = 2, \
        alpha = 0.3, gamma = 0.9, exr = 0.9998, excr = 0.9998) 		   	  			  	 		  		  		    	 		 		   		 		  
    episodes = 800   	
    q_diffs_explore = train_lock_learner(sequence, episodes, learner)		  	 		  		  		    	 		 		   		 		  		
    Pi_star = np.zeros([state_count, 1])   
    for i in range(0, Pi_star.shape[0]):
        Pi_star[i][0] = np.argmax(learner.q[i])  		
    print(learner.q)
    print('This is the policy given by Q Learner Explorer!')		
    printsequence(Pi_star)
    print('')	
    #print(np.round_(Pi_star))

    
    # Q EXPLOIT
    learner = ql.QLearner(num_states=state_count, num_actions = 2, \
        alpha = 0.3, gamma = 0.9, exr = 0.6, excr = 0.9998) 		   	  			  	 		  		  		    	 		 		   		 		  
    episodes = 800   	
    q_diffs_exploit = train_lock_learner(sequence, episodes, learner)		  	 		  		  		    	 		 		   		 		  		
    Pi_star = np.zeros([state_count, 1])   
    for i in range(0, Pi_star.shape[0]):
        Pi_star[i][0] = np.argmax(learner.q[i])  			
    print('This is the policy given by Q Learner Exploiter!')		
    printsequence(Pi_star)
    print('')	

    
    # VALUE LEARNER
    state_count = sum(sequence) + 2
    R_f = np.zeros([state_count , 1])
    R_f[-1,0] = minor_reward
    R_f[-2,0] = major_reward

    learner = vl.ValueIteration(gamma=0.9, num_states=state_count, num_actions=2, R_f = R_f, \
         return_max_V = return_max_V_lock, env = sequence)
    episodes = 20
    diffs_value = np.zeros((episodes,1))
  #  for episode in range(0,episodes):
    diff = -1
    episode = 0
    while diff != 0:
        V_old = learner.V_star.copy()
        learner.iterate_V()
        V_new = learner.V_star.copy()
        diff = np.absolute((V_new - V_old)).max()
        diffs_value[episode,0] = diff
        episode += 1
    policy = learner.return_policy()
    print('This is the policy given by Value Iterationer!')		
    printsequence(policy)
    print('')

    # POLICY LEARNER
    learner = pl.PolicyIteration(gamma=0.9, num_states=state_count, num_actions=2, R_f = R_f, \
         return_max_V = return_max_V_lock, env = sequence)
    episodes = 20
    diffs_policy = np.zeros((episodes,1))
    for episode in range(0,episodes):
        P_old = learner.Pi_star.copy()
        learner.iterate_P()
        P_new = learner.Pi_star.copy()
        diff = np.absolute((P_new - P_old)).sum()
        diffs_policy[episode,0] = diff
    policy = learner.Pi_star
    print('This is the policy given by Policy Iterationer!')		
    printsequence(policy)
    print('')    

    # PLOT RESULTS
    fig, axes = plt.subplots(2,2)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.35)

    ax = axes[1,0]
    ax.plot(q_diffs_explore)
    ax.title.set_text('Exploring Q Learner (explore rate = 0.9998)')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max abs. update on Q table')

    ax = axes[1,1]
    ax.plot(q_diffs_exploit)
    ax.title.set_text('Exploiting Q Learner (explore rate = 0.6)')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max abs. update on Q table')

    ax = axes[0,0]
    ax.plot(diffs_value)
    ax.title.set_text('Value Iteration')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max abs. update on Value table')

    ax = axes[0,1]
    ax.plot(diffs_policy)
    ax.title.set_text('Policy Iteration')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max abs. update on Policy table')
    plt.show()	   	  			  	 		  		  		    	 		 		   		 	

if __name__ == "__main__" :
    import argparse
    print("Running Reinforecement experiments ...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='maze')

    args = parser.parse_args()
    problem = args.problem
    func_eval_count = 0
    if problem == 'maze':
        print("Running Maze Problem:...")
        run_maze_problem('map_large.csv',40)
    if problem == 'maze_random_start':
        print("Running Maze Problem:...")
        run_maze_problem('map_large.csv',40, random_start = True)
    if problem == 'lock':
        print("Running Lock Problem:...")
        run_lock_problem()
