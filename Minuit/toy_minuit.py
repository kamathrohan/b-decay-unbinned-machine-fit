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

FIX=[
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


class toy:


    def __init__(self , model='SM'):

        self.model = model
        self.coeffs = [] #Model coeffs
        self.events = [] #generated events
        self.coeff_fit = [] #fitted coefficients 
        self.NLL0=0
        self.NLL=0
        self.FIX=[
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
    def get_coeffs(self):
        return self.coeffs
    def generate(self, events = 2400,verbose = False):
        
        if self.model == "SM":
            signal_coeffs = bmf.coeffs.signal(bmf.coeffs.SM)
            self.coeffs =[i.numpy() for i in signal_coeffs] 
            if verbose:  
                print("Ideal coeffs for SM:", self.coeffs )
            t0=time.time()
            self.events = bmf.signal.generate(signal_coeffs, events_total=events).numpy()
            t1=time.time()
            self.NLL0=self.nll_iminuit(self.coeffs)
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


    
    def minuitfit(self, Ncall=2400, init= 'DEFAULT' , fixed=None , coefini=None , verbose = False):

        #if init == None or init == 'DEFAULT': 
            #A=bmf.coeffs.fit(initialization= FIT_INIT_TWICE_LARGEST_SIGNAL_SAME_SIGN )
        A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_default , current_signal_model=self.model)
        if fixed is None : fixed = self.FIX
        '''
        elif init == 'FIXED' :
            #A=bmf.coeffs.fit(initialization= FIT_INIT_CURRENT_SIGNAL )

        
            A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_fixed)
        elif init == 'ANY_SIGN' : 
            #A=bmf.coeffs.fit(initialization= FIT_INIT_TWICE_CURRENT_SIGNAL_ANY_SIGN )

            A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_any_sign)

        '''

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

        
FIX0=[
0,1,1,
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
1,1,1
]

FIX1=[
1,0,0,
0,0,0,
0,0,0,
0,0,0,
0,0,0,
0,0,0,
0,0,0,
0,0,0,
0,0,0,
0,0,0,
0,0,0,
0,0,0,
0,0,0,
0,0,0,
0,0,0,
0,0,0,
]



'''
x=[]
NLL_profile=[]
A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_default , current_signal_model='SM')
X=[A[i].numpy() for i in range(len(A))]
e= 10**(1)
X[0] -= 5*e

toy1 = toy( model='SM')

toy1.generate( verbose = False )

for j in tqdm(range(10)):
    X[0] += e 
    

    coef = toy1.minuitfit(Ncall=10000 , verbose=True , coefini=X , fixed=FIX1)
    nll0=toy1.NLL0
    nll=toy1.NLL

    print('Initial NLL : ' , nll0.numpy())


    print('Final NLL : ' , nll.numpy())

    NLL_profile.append(nll.numpy())
    x.append(X[0])


print(NLL_profile)
print(x)

fig, ax = plt.subplots()
plt.plot( x , NLL_profile )
ymin, ymax = ax.get_ylim()
#ax.vlines(  , ymin , ymax , label='MC value')
plt.show()



'''