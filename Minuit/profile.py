import numpy as np 
import matplotlib.pyplot as plt 
import b_meson_fit as bmf 
from toy_minuit import toy
import time 
from tqdm import tqdm
import csv 
from test_iminuit import amplitude_latex_names ,amplitude_names, LaTex_labels , Standard_labels
from scipy import interpolate 


LaTex=LaTex_labels(amplitude_latex_names)
Title=Standard_labels(amplitude_names)

fix_array=[
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


toy1 = toy( model='SM')

toy1.generate( events = 10000 , verbose = True )
A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_default , current_signal_model='SM')
Coeff0=[A[i].numpy() for i in range(len(A))]
m , coef = toy1.minuitfit(Ncall=1000 , verbose=False , coefini=Coeff0 , fixed=fix_array)


B=[m.errors[i] for i in m.errors]
B=np.array(B)

print(B)


n=11
J=0

for j in range(0 , 48):

    if fix_array[j]==0:

        #print('\n' , 'Simulation #' , j+1 , '\n' , amplitude_names[j], '\n')
        
        fix=fix_array[:]
        X=toy1.coeffs[:]
        fix[j]=1
        x=[]
        NLL_profile=[]
        nll0=toy1.NLL0
        
        e = B[j]
        X[j] -= np.ceil(n/2)*e       
        
        print('Fixed parameters : ' , fix )
        print('Initial coeffs : ' , X)

        print('Param ini : ' , X[j] , ' with errors : ' , B[j])
        print('Initial NLL (normalised) : ' , nll0.numpy() , '\n')



        for i in range(n):
            X[j] += e 
            m , coef = toy1.minuitfit(Ncall=100 , verbose=False , coefini=X , fixed=fix)
            nll=toy1.NLL

            
            if i == 5 : 
                print('Profile param (central) : ' , X[j] , '\n'  , 'Final NLL (normalised) : ' , nll.numpy() )
            else : 
                print('Profile param :' , X[j] , '\n'  , 'Final NLL (normalised) : ' , nll.numpy())

            NLL_profile.append(nll.numpy())
            x.append(X[j])


        
        fig, ax = plt.subplots()
        NLL_profile=np.array(NLL_profile)
        y=NLL_profile-min(NLL_profile)
        f = interpolate.interp1d(x, y)
        X=np.linspace(x[0] , x[-1])
        print(X)
        plt.plot(X , f(X) , 'r-.' , label='Parabolic interp.')
        plt.plot( x , y  , 'k.' , label='Profile nll')
        plt.plot(x[int(np.floor(11/2))] , y[int(np.floor(11/2))] , 'ro'  , label='MC value')
        plt.plot(x[np.argmin(y)] , min(y) , 'ko'  , label='Migrad value')
        
        ax.set_title('Expected Value:'+str(Coeff0[j])+' Converged Value:'+str(x[np.argmin(y)] ))
        ymin, ymax = ax.get_ylim()
        #ax.vlines( toy1.coeffs[j]  , ymin , ymax , label='MC value')
        #plt.show()
        ax.fill_between(X, f(X), 0.5 , where=0.5 > f(X) , alpha=0.5, color='red')
        ax.legend()
        #ax.axhspan(ymin, 0.5, alpha=0.1, color='red' , label= r'$\pm \sigma$')
        path='./Minuit/Plot_profiles/'

        plt.savefig(path+Title[j]+'.png')

        bol=0
        if bol==1 : 
            with open(r"./Minuit/Test_stats/data_profile.csv", 'a') as data:
                writer =csv.writer(data)
                writer.writerow(NLL_profile)
                writer.writerow(x) 
            data.close()
