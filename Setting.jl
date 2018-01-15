global Numb_Operators = 3
# global Numb_Operators = 4
global Numb_BS = 6
global dim_x = 10
global dim_y = 10
global Q  = 865 #Watts
global B  = 388 # 388 Watts    ### STANDARD rho = 0.31069
global m = 0.2
global Dx = 100   #100mx100m
global Dy = Dx

# global T_req = 16    # Time horizon
# global dt= 2.        # 30
# global T_req = 8     # Time horizon
global T_req = 6       # Time horizon
# global dt= 1.          # 1 hour
global dt= 1/T_req     # 1 hour

# global BS = [ 200 150;
#               150 850;
#               450 500;
#               800 150;
#               850 870]
# global increased_arr_rate=0.5
# global lambs = [6.8 5.7 6.1]

global BS = [ 150 100;
              150 850;
              300 500;
              700 500;
              800 150;
              850 900]

global increased_arr_rate= 4.4
global lambs = [5.2 19.1 0.11]   # limited power
# global lambs = [6. 18.5 1.]   # limited power
# global lambs = [4.16 15.3 0.41]   # limited energy
# global increased_arr_rate= 11.
# global lambs = [20. 42. 10.]   # No limited power
# global lambs = [3.48 9.9 .73]    #8 steps MPC
# global lambs = [3.4 3.5 0.64 0.67]


global REUSED_TRAFFIC = true
global MPC_REUSED_TRAFFIC = true
global MPC_REUSED_READ_ONLY = true
global DEBUG = 1 #LEVEL 0, 1, 2, 3
global TRAFFIC_CONTROL = 1      # 0, 1, 2:  Low, Medium, High
global RUNNING_MODE = 2   #1 , 2, 3:   Pure MPC, Optimial MPC, Greedy Comparison
global DEFERRABLE_MODE = false
global ALPHA_OBJECTIVE = 2

global CENTRALIZED_BS = false
global CEN_B = 500 # limited power at BS 3

global ERROR_CODE = 0
global OK_CODE = 1
global NOT_CONVERGENCE = 2
