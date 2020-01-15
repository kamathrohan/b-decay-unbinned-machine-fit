import numpy as np  
import tensorflow as tf
import b_meson_fit as bmf 
import b_meson_fit.coeffs as bmfc
import b_meson_fit.signal as bmfs
import time 
from iminuit import Minuit 
import matplotlib.pyplot as plt 
#Define names #


#Define boolean sequence of variables to keep fixed 

#1,1,1,
fix_array=[0,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,]
print(fix_array)

def nll(signal_coeffs):
    """Get the normalized negative log likelihood

    Working with the normalised version ensures we don't need to re-optimize hyper-parameters when we
    change signal event numbers.

    Returns:
        Scalar tensor
    """
    return bmfs.normalized_nll(signal_coeffs, signal_events)

def set_coef(Coef_INIT , Coef0,  fix_array ) :
    for i in range(len(Coef_INIT)):
        if fix_array[i]==1 :
            Coef_INIT[i]=  Coef0[i]
    return Coef_INIT



##   Initialize coefficients 
signal_coeffs = bmf.coeffs.signal(bmf.coeffs.SM)
Coef0=[i.numpy() for i in signal_coeffs]
print("Ideal coeffs for SM:", Coef0 )

A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_default)
Coef_INIT=[A[i].numpy() for i in range(len(A))]


##   Generate data to fit 
t0=time.time()
signal_events = bmf.signal.generate(signal_coeffs, events_total=24000)
t1=time.time()
print("Time taken to generate data:", t1-t0)

##   Perform minuit fit 
Coef_INIT=set_coef(Coef_INIT, Coef0, fix_array)
print("Initial coeffs for minuit fit:", Coef_INIT)
m = Minuit.from_array_func(nll,Coef_INIT, fix= fix_array , errordef=1, pedantic = False)
t0=time.time()
m.migrad( ncall=1000)
t1=time.time()
print("Time taken to fit :", t1-t0)


#Print fit results 
Coef_OUT=[m.values[i] for i in range(len(m.values))]
print(Coef_OUT)
print('NLL for final set of coeffs:' , nll(Coef_OUT))


#Plot nll profile fr given parameters
m.profile(m.parameters[0])
m.draw_profile(m.parameters[0])
#plt.xlabel()
#plt.ylabel()
plt.show()

