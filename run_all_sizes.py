import numpy as np
import importlib
import run_one_size
importlib.reload(run_one_size)   #impose if there would be changing in run_one_size so  reload if edited
import os
from run_one_size import run_for_L

L_list = [20,30,40,50,60,80]
Tc_results = []
for N in L_list:
    Tc_chi, Tc_cv = run_for_L(N)
    Tc_results.append([N, Tc_chi, Tc_cv])

print("\nAll sizes completed.\n")
print("Results ( L , Tc_chi , Tc_cv ):")
for row in Tc_results:
    print(row)
