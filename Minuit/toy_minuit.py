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
    def __init__(self , model='SM' ):

        self.model = model
        self.events_bool = False
        self.coeffs = [] #Model coeffs
        self.events = [] #generated events
        self.coeff_fit = [] #fitted coefficients 
        self.coeff_init = []
        self.NLL0=0
        self.NLL=0
        self.FIX=fix_array

        
    def get_coeffs(self):
        return self.coeffs


    def generate(self, events = 2400 ,  frac_background= None , coeff_background=None, verbose = False):
        
        if self.model == "SM" :
            signal_coeffs = bmf.coeffs.signal(bmf.coeffs.SM)
            self.coeffs =[i.numpy() for i in signal_coeffs] 
        elif self.model == 'NP' :
            signal_coeffs = bmf.coeffs.signal(bmf.coeffs.NP)
            self.coeffs =[i.numpy() for i in signal_coeffs] 
        else :
            raise ValueError('current signal model unknown')

        if verbose:  
            print("Ideal coeffs for ", self.model , ' : ', self.coeffs )
        
        t0=time.time()
        if frac_background is not None and sum(len(CoeffBCK) for CoeffBCK in coeff_background)== 8 : 
            n_bck = int(np.round(events*frac_background))
            n_signal = int(events - n_bck)
            signal = bmfs.generate(signal_coeffs, events_total=n_signal, batch_size=10_000_000)   
            bck = bmfs.generate_background(coeff_background ,  events_total=n_bck, batch_size=10_000_000)
            self.events = tf.concat([signal , bck] , 0)[0:events, :]
        elif frac_background is not None and not isinstance(frac_background , float)  :  
            raise ValueError('frac_background must be a float')
        else : 
            self.events = bmfs.generate(self.coeffs , events_total=events, batch_size=10_000_000)

        
        t1=time.time()
        self.NLL0=self.nll_iminuit(self.coeffs)
        self.events_bool = True
        
        if verbose:
            print("Time taken to generate data:", t1-t0)
        
    
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

    def tf_fit(self, Ncall=None, init= 'DEFAULT' , fixed=None , coefini=None , verbose = False , opt_params=None):
        if init ==None or init == 'DEFAULT' : 
            A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_default , current_signal_model= self.model , fix=fixed)
        
        elif init == 'SAME SIGN' : 
            A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_same , current_signal_model=self.model , fix=fixed)
            
        elif init == 'ANY SIGN' :
            A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_any , current_signal_model=self.model , fix=fixed)
        
        if coefini is not None  :
            A=bmf.coeffs.fit(initialization=coefini , fix=fixed)        
        events=tf.convert_to_tensor(self.events)


        if verbose:
            print('\n', "Coeffs used for MC:", self.coeffs)
            print("Initial coeffs for tensorflow fit:", [A[j].numpy() for j in range(len(A))])

        self.coeff_init = [A[i].numpy() for i in range(len(A))]
        optimizer = bmf.Optimizer( A, events, opt_name='AMSGrad', learning_rate=0.20, opt_params=opt_params )  
        converged = False
        j=0
        t0=time.time()
        while converged == False :
            optimizer.minimize()
            if Ncall is not None and j>Ncall:
                tfCoeff=[optimizer.fit_coeffs[i].numpy() for i in range(len(optimizer.fit_coeffs))]
                self.coeff_fit=tfCoeff
                self.NLL=self.nll_iminuit(self.coeff_fit)
                return optimizer , tfCoeff

            j+=1
            if optimizer.converged() == True :
                converged= True 
        t1=time.time()            

        tfCoeff=[optimizer.fit_coeffs[i].numpy() for i in range(len(optimizer.fit_coeffs))]
        self.coeff_fit=tfCoeff
        self.NLL=self.nll_iminuit(tfCoeff)
        if verbose:
            print('\n', ' Fitted coefficients : ' , self.coeff_fit)
            print( '\n', "Time taken to fit :", t1-t0)
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

        if coefini is not None and len(coefini)==48 :
            A=bmf.coeffs.fit(initialization=coefini , current_signal_model='SM' , fix = fixed )
            Coef_INIT=[A[i].numpy() for i in range(len(A))]
        
        else : 
            Coef_INIT=[A[i].numpy() for i in range(len(A))]
            Coef_INIT= self.set_coef(Coef_INIT, self.coeffs , fixed)

        if verbose:
            print('\n', "Coeffs used for MC:", self.coeffs)
            print("Initial coeffs for minuit fit:", Coef_INIT)
        self.coeff_init = Coef_INIT
        m = Minuit.from_array_func(self.nll_iminuit,Coef_INIT, fix= fixed , errordef=0.5, pedantic = False)
        
        t0=time.time()
        m_final=m.migrad( ncall=Ncall )


        t1=time.time()            
        ##   Print fit results 
        self.coeff_fit=[m.values[i] for i in range(len(m.values))]
        self.NLL=self.nll_iminuit(self.coeff_fit)
        if verbose:
            print('\n', ' Fitted coefficients : ' , self.coeff_fit)
            print( '\n', "Time taken to fit :", t1-t0)
        return m , self.coeff_fit 


'''
A=[]
B=[]

A0=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_default , current_signal_model='SM')
Coeff0=[A0[i].numpy() for i in range(len(A0))]
Coeff0[0]+=4

toy1=toy('SM')
toy1.generate(2400 , verbose=False )
test_tf=[10 , 100 , 1000 , 10000 , 100000]
FIX=fix_alphas
FIX[0]= 1
C=bmf.coeffs.fit(Coeff0  , fix=fix_alphas)
print(C)
toy1.tf_fit()



for j in range(len  (test_tf)):
    


    m , coef= toy1.tf_fit(test_tf[j] , coefini=inii , fixed=fix_one_alpha)
    print(toy1.NLL)
    print(coef[0])
    B.append(coef[0])


for i in tqdm(range(3)):
    C=bmf.coeffs.fit(bmf.coeffs.fit_initialization_same , current_signal_model='SM' , fix=fix_one_alpha)
    print(C)
    m , migradCoeff=toy1.minuitfit(10000 , coefini= C ,fixed=fix_one_alpha)
    optimizer , tfCoeff =toy1.tf_fit( coefini =C ,  fixed=fix_one_alpha )
    A.append(tfCoeff[0])
    B.append(migradCoeff[0])
    print(tfCoeff[0] , migradCoeff[0] )

print( 'TF Mean is :' , np.mean(A) , 'pm' , np.std(A))
print( ' Migrad Mean is :' , np.mean(B) , 'pm' , np.std(B))
'''

