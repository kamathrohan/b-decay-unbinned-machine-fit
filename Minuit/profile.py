import numpy as np 
import matplotlib.pyplot as plt 
import b_meson_fit as bmf 
from toy_minuit import toy
import time 
from tqdm import tqdm
import csv 
from test_iminuit import amplitude_latex_names ,amplitude_names, LaTex_labels , Standard_labels  , fix_array
from scipy import interpolate 
from textwrap import wrap


LaTex=LaTex_labels(amplitude_latex_names)
Title=Standard_labels(amplitude_names)

toy1 = toy( model='SM')

toy1.generate( events = 10000 , verbose = False )
A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_default , current_signal_model=toy1.model)
Coeff0=[A[i].numpy() for i in range(len(A))]
m , coef = toy1.minuitfit(Ncall=100 , verbose=False , coefini=Coeff0 , fixed=fix_array)


B=[m.errors[i] for i in m.errors]
B=np.array(B)

print(B)

path='./Minuit/Plot_profiles/TryIt/'
n=11
J=0
idx=48

for j in tqdm(range(0 , idx)):

    if fix_array[j]==0:
        
        fix=fix_array[:]
        fix[j]=1
        x=[]
        NLL_profile100=[]       
        NLL_profile1000=[]
        NLL_profile10000=[]
        nll0=toy1.NLL0
        
        e = B[j]

        A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_fixed0 , current_signal_model=toy1.model)
        X=[A[i].numpy() for i in range(len(A))]
        X[j]=Coeff0[j]
        X[j] -= np.ceil(n/2)*e

        for i in range(n):
            X[j] += e 
            m_100 , coef_100 = toy1.minuitfit(Ncall=100 , init= 'SAME SIGN' ,verbose=False , coefini=X , fixed=fix)
            nll100=toy1.NLL
            m_1000 , coef_1000 = toy1.minuitfit(Ncall=1000 , init= 'SAME SIGN' ,verbose=False , coefini=X , fixed=fix)
            nll1000=toy1.NLL
            m_10000 , coef_10000 = toy1.minuitfit(Ncall=10000 ,init= 'SAME SIGN' , verbose=False , coefini=X , fixed=fix)
            nll10000=toy1.NLL

            print(X[j] , ' : ' , nll100.numpy()/10000 , '&' , nll1000.numpy()/10000 , '&' , nll10000.numpy()/10000 )

            NLL_profile100.append(nll100.numpy())
            NLL_profile1000.append(nll1000.numpy())
            NLL_profile10000.append(nll10000.numpy())

            x.append(X[j])

        fig, ax = plt.subplots()

        NLL_profile100=np.array(NLL_profile100)
        NLL_profile1000=np.array(NLL_profile1000)
        NLL_profile10000=np.array(NLL_profile10000)

        #y100=NLL_profile100-min(NLL_profile100)
        y100=NLL_profile100/10000
        #y1000=NLL_profile1000-min(NLL_profile1000)
        y1000=NLL_profile1000/10000
        #y10000=NLL_profile10000-min(NLL_profile10000)
        y10000=NLL_profile10000/10000

        f100 = interpolate.interp1d(x, y100)
        f1000 = interpolate.interp1d(x, y1000)
        f10000 = interpolate.interp1d(x, y10000)

        X=np.linspace(x[0] , x[-1])
        plt.plot(X , f100(X) , 'r-.' , label='Migrad100')
        plt.plot(X , f1000(X) , 'b-.' , label='Migrad1000')
        plt.plot(X , f10000(X) , 'g-.', label='Migrad10000')


        plt.plot( x , y100  , 'k.' )
        plt.plot( x , y1000  , 'k.' )
        plt.plot( x , y10000  , 'k.' )


        plt.plot(x[int(np.floor(11/2))] , y100[int(np.floor(11/2))] , 'cD'  , label='MC value')
        plt.plot(x[int(np.floor(11/2))] , y1000[int(np.floor(11/2))] , 'cD' )
        plt.plot(x[int(np.floor(11/2))] , y10000[int(np.floor(11/2))] , 'cD' )

        plt.plot(x[np.argmin(y100)] , min(y100) , 'ro' )
        plt.plot(x[np.argmin(y1000)] , min(y1000) , 'bo' )
        plt.plot(x[np.argmin(y10000)] , min(y10000) , 'go'  )


        title = ax.set_title('Expected Value:'+str(format( Coeff0[j], '.3f'))+'      Migrad100:' +str(format( x[np.argmin(y100)], '.3f'))+ '\n' + 'Migrad1000:' +str(format( x[np.argmin(y1000)], '.3f'))+ '      Migrad10000:' +str(format( x[np.argmin(y10000)], '.3f')) )
        #+r'$\pm$'+str(format(B[j], '.4f')) +
        ymin, ymax = ax.get_ylim()
        #ax.vlines( toy1.coeffs[j]  , ymin , ymax , label='MC value')
        #plt.show()
        
        #ax.fill_between(X, f100(X), 0.5 , where=0.5 > f100(X) , alpha=0.5, color='red')
        #ax.fill_between(X, f1000(X), 0.5 , where=0.5 > f1000(X) , alpha=0.5, color='blue')

        ax.legend()
        ax.set(xlabel=LaTex[j], ylabel=r' NLL')        #ax.axhspan(ymin, 0.5, alpha=0.1, color='red' , label= r'$\pm \sigma$')        
        plt.subplots_adjust(top=0.9)
        plt.savefig(path+Title[j]+'.png')
