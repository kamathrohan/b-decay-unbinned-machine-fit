import numpy as np  
import tensorflow as tf
import b_meson_fit.coeffs as bmfc
import b_meson_fit.signal as bmfs
import b_meson_fit as bmf 
import time 
from iminuit import Minuit 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import math


class toy:
    def __init__(self):
        self.model = None
        self.coeffs = [] #Model coeffs
        self.events = [] #generated events
        self.fix_array = [
                    0,0,0,
                    0,0,0,
                    0,0,0,
                    0,0,0,
                    0,0,0,
                    0,0,0,
                    0,0,0,
                    1,1,1,
                    0,0,0,
                    1,1,1,
                    1,1,1,
                    1,1,1,
                    0,1,1,
                    0,1,1,
                    0,1,1,
                    0,1,1,
                    ]
        self.coeff_fit = [] #fitted coefficients 

    def generate(self, events = 24000, model = "SM",verbose = False):
        if model == "SM":
            signal_coeffs = bmf.coeffs.signal(bmf.coeffs.SM)
            self.coeffs =[i.numpy() for i in signal_coeffs] 
            if verbose:  
                print("Ideal coeffs for SM:", self.coeffs )
            t0=time.time()
            self.events = bmf.signal.generate(signal_coeffs, events_total=events).numpy()
            t1=time.time()
            if verbose:
                print("Time taken to generate data:", t1-t0)
        return

    def nll_iminuit(self,signal_coeffs):
        return bmfs.nll(signal_coeffs,self.events)
    
    def set_coef(self, Coef_INIT , Coef0,  fix_array ):
        for i in range(len(Coef_INIT)):
            if fix_array[i]==1 :
                Coef_INIT[i]=  Coef0[i]
        return Coef_INIT

    def minuitfit(self, Ncall=10000,verbose = False):
        if len(self.events) == 0:
            print("No events generated")
            return
        A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_default)
        Coef_INIT=[A[i].numpy() for i in range(len(A))]
        Coef_INIT= self.set_coef(Coef_INIT, self.coeffs, self.fix_array)

        if verbose:
            print("Coeffs used for MC:", self.coeffs)
            print("Initial coeffs for minuit fit:", Coef_INIT)

        m = Minuit.from_array_func(self.nll_iminuit,Coef_INIT, fix= self.fix_array , errordef=0.5, pedantic = False)
        t0=time.time()
        m_final=m.migrad( ncall=Ncall)
        #if m_final.is_above_max_edm==True :
         #   print('NOOOOOOOOOOOOO')
        print(m.migrad_ok())
        print(type(m_final))
        print(m_final )
        t1=time.time()
        if verbose:
            print("Time taken to fit :", t1-t0)
        ##   Print fit results 
        self.coeff_fit=[m.values[i] for i in range(len(m.values))]
        if verbose:
            print(self.coeff_fit)
            print('NLL for final set of coeffs:' , self.nll_iminuit(self.coeff_fit))
        return

        
 





    

'''
toy1 = toy()
t0=time.time()
toy1.generate()
t1=time.time()
print(t1-t0)
t0=time.time()
toy1.minuitfit(Ncall=100)
t1=time.time()
print(t1-t0)
print(toy1.coeff_fit)
'''