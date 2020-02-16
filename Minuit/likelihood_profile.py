import numpy as np 
import matplotlib.pyplot as plt 
import b_meson_fit as bmf 
from toy_minuit import toy
import time 
from tqdm import tqdm
import csv 
from test_iminuit import amplitude_latex_names ,amplitude_names, LaTex_labels , Standard_labels  , fix_array , fix_alphas , fix_one_alpha,fix_alpha_beta, coeff_any_sign,coeff_default,coeff_same_sign
from scipy import interpolate 
from textwrap import wrap

LaTex=LaTex_labels(amplitude_latex_names)
Title=Standard_labels(amplitude_names)
model = "SM"

toy1 = toy(model='SM')
print("Time Start:", time.ctime())
toy1.generate(events = 1000000)
#events = np.loadtxt("./Minuit/toy.csv")
#toy1.populate_events(events)
fix = fix_alphas

def initial(coeffs,fixed):
    coefini = []
    for i in range(48):
        if fix_alphas[i]==0:
            coefini.append(coeff_same_sign[i])
        else:
            coefini.append(coeff_default[i])
    return coefini


def pulls(m,fix, coeff_default):
    pulls = []
    for i in range(48):
        if fix[i] == 0:
            pulls.append((m.values[i]-coeff_default[i])/m.errors[i])
    return pulls 

    
#np.savetxt("toy_10M.csv",toy1.get_events())
t1 = time.ctime()
print("Time generate: ",t1)
m , coef = toy1.minuitfit(Ncall=1000000 , verbose=False , coefini=initial(coeff_same_sign,fix_alpha_beta) , fixed=fix_alpha_beta)
t2 = time.ctime()
print("Time Finish: ",t2)
print(pulls(m,fix_alpha_beta, coeff_default))

