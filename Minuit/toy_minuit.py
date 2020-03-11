import numpy as np  
import tensorflow as tf
import b_meson_fit.coeffs as bmfc
import b_meson_fit.signal as bmfs
import b_meson_fit as bmf 
from test_iminuit import amplitude_latex_names ,amplitude_names, LaTex_labels , Standard_labels  , fix_array , fix_alphas , fix_one_alpha , fix_alpha_beta , fix_alpha_beta_gamma1 
import time 
from iminuit import Minuit 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import math




class toy:
    def __init__(self , model='SM'):

        self.model = model
        self.events_bool = False
        self.coeffs = [] #Model coeffs
        self.events = [] #generated events
        self.coeff_fit = [] #fitted coefficients 
        self.NLL0=0
        self.NLL=0
        self.FIX=fix_array
        
    def get_coeffs(self):
        return self.coeffs

    def generate(self, events = 2400,verbose = False ):

        if self.model == "SM":
            signal_coeffs = bmf.coeffs.signal(bmf.coeffs.SM)
            self.coeffs =[i.numpy() for i in signal_coeffs] 
            if verbose:  
                print("Ideal coeffs for SM:", self.coeffs )
            t0=time.time()
            self.events = bmf.signal.generate(signal_coeffs, events_total=eventspoisson).numpy()
            t1=time.time()
            self.NLL0=self.nll_iminuit(self.coeffs)
            self.events_bool = True
            if verbose:
                print("Time taken to generate data:", t1-t0)

        elif self.model == "NP":
            signal_coeffs = bmf.coeffs.signal(bmf.coeffs.NP)
            self.coeffs =[i.numpy() for i in signal_coeffs] 
            if verbose:  
                print("Ideal coeffs for NP:", self.coeffs )
            t0=time.time()
            self.events = bmf.signal.generate(signal_coeffs, events_total=events).numpy()
            t1=time.time()
            self.NLL0=self.nll_iminuit(self.coeffs)
            self.events_bool = True
            if verbose:
                print("Time taken to generate data:", t1-t0)
          

        else :
            raise ValueError('current signal model unknown')

    def nll_iminuit(self,signal_coeffs):
        return bmfs.nll(signal_coeffs,self.events)

    def set_coef(self, Coef_INIT , Coef0,  fix=None ):
        if fix is None : fix=self.FIX
        for i in range(len(Coef_INIT)):
            if fix[i]==1 :
                Coef_INIT[i]=  Coef0[i]
        return Coef_INIT

    def get_events(self):
        if not self.events_bool:
            raise ValueError("Generate Events first")
        else:
            return self.events

    def populate_events(self,events):
        if not self.events:
            self.events = events
            return
        else:
            raise ValueError("Events already generated!")

    def tf_fit(self, Ncall=None, init= 'DEFAULT' , fixed=None , coefini=None , verbose = False):
        
        if init ==None or init == 'DEFAULT' : 
            A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_default , current_signal_model= self.model , fix=fixed)

        elif init == 'SAME SIGN' : 
            A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_same , current_signal_model=self.model , fix=fixed)
            
        elif init == 'ANY SIGN' :
            A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_any , current_signal_model=self.model , fix=fixed)
        
        events=tf.convert_to_tensor(self.events)
        optimizer = bmf.Optimizer(
            A,
            events,
            opt_name='AMSGrad',
            learning_rate=0.20,
            opt_params=None,
            grad_clip=None,
            grad_max_cutoff=5e-6
        )      
        converged = False
        j=0
        while converged == False :
            optimizer.minimize()
            if Ncall is not None :
                if j>Ncall :
                    tfCoeff=[optimizer.fit_coeffs[i].numpy() for i in range(len(optimizer.fit_coeffs))]
                    return optimizer , tfCoeff

            j+=1
            if optimizer.converged() == True :
                converged= True 

        tfCoeff=[optimizer.fit_coeffs[i].numpy() for i in range(len(optimizer.fit_coeffs))]
        return optimizer , tfCoeff 
    
    def minuitfit(self, Ncall=10000, init= 'DEFAULT' , fixed=None , coefini=None , verbose = False):

        #if init == None or init == 'DEFAULT': 
            #A=bmf.coeffs.fit(initialization= FIT_INIT_TWICE_LARGEST_SIGNAL_SAME_SIGN )
        if init ==None or init == 'DEFAULT' : 
            A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_default , current_signal_model=self.model)

        elif init == 'SAME SIGN' : 
            A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_same , current_signal_model=self.model)

        elif init == 'ANY SIGN' :
            A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_any , current_signal_model=self.model)

        if fixed is None : fixed = self.FIX

        if len(self.events) == 0:
            print("No events generated")
            return
        

        if coefini is not None :
            Coef_INIT=coefini
        else : 
            Coef_INIT=[A[i].numpy() for i in range(len(A))]
            Coef_INIT= self.set_coef(Coef_INIT, self.coeffs , fixed)

        if verbose:
            print('\n', "Coeffs used for MC:", self.coeffs)
            print("Initial coeffs for minuit fit:", Coef_INIT)

        m = Minuit.from_array_func(self.nll_iminuit,Coef_INIT, fix= fixed , errordef=0.5, pedantic = False)
        t0=time.time()
        m_final=m.migrad( ncall=Ncall)


        t1=time.time()
        if verbose:
            print("Time taken to fit :", t1-t0)
        ##   Print fit results 
        self.coeff_fit=[m.values[i] for i in range(len(m.values))]
        self.NLL=self.nll_iminuit(self.coeff_fit)
        if verbose:
            print('\n', ' Fitted coefficients : ' , self.coeff_fit)
        return m , self.coeff_fit 
