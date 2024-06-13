from .kubecalls import *
from .localcalls import *

iit_working_cases = [3, 8, 24, 16, 13, 11, 21, 38] # 4: not working for 110
# working_cases = [16, 8, 12, 3, 7, 25, 1, 24, 21, 18, 13, 11, 19, 38, 4, 14, 6]
# iit_not_working_cases = [12, 7, 25, 
#                          18, # 100 with lr 1e-3
#                          14, # ~99
#                          6, # 97
#                          19]
new_working_cases = [4, 14, 18, 19, 33, 34, 35, 36, 37, 18, 38, 16, 20, 26, 29]
still_running = [38, 16]
not_working = [2, 5, 39, 9, 10, 15, 17, 22, 23, 30, 31, 1, 25, 28, 7,]
# working_cases = [14, 18,] # ]
working_cases = list((set(new_working_cases) - set(still_running)) - set(not_working))
# working_cases = [24]
to_check_cases = [27, 6, 12, 32]
all_working_cases = list(set(working_cases + iit_working_cases) - set(still_running))
# len(working_cases) + len(not_working) + len(iit_working_cases) + len(to_check_cases)