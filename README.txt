Steps to run the code:
1. Code link: https://github.com/tri47/reinforcement_learning.git
2. Set up the environment with the libraries in enviroment.yml file (using conda or pip)
The libraries include:
- python=3.7.2
- matplotlib=3.0.3
- numpy=1.16.2
- pandas=0.24.1
3. Run the program to run the 2 problems:
    3a. Run the maze (large) problem with no random restarts 
    python reinforcement_learning.py --problem="maze"

    3b. Run the maze (large) problem with random restarts 
    python reinforcement_learning.py --problem="maze_random_start" 

    3c. Run the combination lock problem
    python reinforcement_learning.py --problem="lock"