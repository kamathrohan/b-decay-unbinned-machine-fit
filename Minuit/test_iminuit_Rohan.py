
import argparse
import shutil
import tqdm
import numpy as np
from iminuit import Minuit
import time
import itertools
import b_meson_fit as bmf


def minuitnll(coeffs):
    #newcoeffs = [coeffs[i]/(10**rescaler[i])for i in range(len(coeffs))]
    return bmf.signal.nll(coeffs, signal_events).numpy()


optcoeff = bmf.coeffs.signal(bmf.coeffs.SM)

rescaler = [0,1,0,
3,3,1,
1,3,0,
1,2,1,
0,2,0,
2,3,2,
2,2,0,
0,0,0,
0,1,0,
0,0,0,
0,0,0,
0, 0, 0,
0, 0, 0,
0, 0, 0,
0, 0, 0,
0, 0, 0,]

rescaledcoeffs = [optcoeff[i]*(10**rescaler[i])for i in range(len(optcoeff))]
fit_trainable_idxs = list(itertools.chain(range(0, 21) ,range(24,27), [36], [39], [42], [45]))

fix = []

for i in range(48):
    if i in fit_trainable_idxs:
        fix.append(0)
    else:
        fix.append(1)




signal_coeffs = bmf.coeffs.signal(model = "SM")

signal_events = bmf.signal.generate(signal_coeffs, events_total=24000)

t0 = time.time()

m = Minuit.from_array_func(minuitnll,rescaledcoeffs, fix = fix, errordef=0.5, pedantic = False)




#m.get_param_states()

m.migrad()

m.hesse()

#print(optcoeff)

t1 = time.time()

print(t1-t0)

for i in range(len(m.values)):
    print(m.values[i],m.errors[i],rescaledcoeffs[i])
