import numpy as np 
import matplotlib.pyplot as plt 
import b_meson_fit as bmf 
from toy_minuit import toy
import time 
from tqdm import tqdm
import csv 
from test_iminuit import amplitude_latex_names ,amplitude_names, LaTex_labels , Standard_labels  , fix_array , fix_alphas , fix_one_alpha, coeff_any_sign,coeff_default,coeff_same_sign
from scipy import interpolate 
from textwrap import wrap

LaTex=LaTex_labels(amplitude_latex_names)
Title=Standard_labels(amplitude_names)
model = "SM"

toy1 = toy( model='SM')
events = np.loadtxt("./Minuit/toy.csv")
toy1.populate_events(events )