import numpy as np 
import matplotlib.pyplot as plt 
import b_meson_fit as bmf 
from toy_minuit import toy
import time 
from tqdm import tqdm
import csv 
from test_iminuit import amplitude_latex_names ,amplitude_names, LaTex_labels , Standard_labels  , fix_array , fix_alphas , fix_one_alpha
from scipy import interpolate 
from textwrap import wrap



LaTex=LaTex_labels(amplitude_latex_names)
Title=Standard_labels(amplitude_names)
init_scheme  = bmf.coeffs.fit_initialization_same #fit_initialization_scheme_default #fit_initialization_any
init_scheme_1= bmf.coeffs.fit_initialization_same
FIX=fix_array #fix_array #fix_alphas
N=10000

toy1 = toy( model='SM')

toy1.generate( events = N , verbose = False )
A0=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_default , current_signal_model=toy1.model)
Coeff0=[A0[i].numpy() for i in range(len(A0))]
m , coef = toy1.minuitfit(Ncall=100 , verbose=False , coefini=Coeff0 , fixed=FIX)


B=[m.errors[i] for i in m.errors]
B=np.array(B)


path='./Minuit/Plot_profiles/TryIt/'
n=11
J=0
idx=48


A=bmf.coeffs.fit(init_scheme , current_signal_model=toy1.model)
A_bis=bmf.coeffs.fit(init_scheme_1 , current_signal_model=toy1.model)

for j in tqdm(range(0 , idx)):

    if FIX[j]==0:
        print(j)
        
        fix=FIX[:]
        fix[j]=1
        x=[]
        NLL_profile=[]
        NLL_profile100=[] 
        NLL_profile200=[] 
        NLL_profile300=[] 
        NLL_profile400=[] 
        NLL_profile500=[]       
        NLL_profile1000=[]
        NLL_profile10000=[]
        nll0=toy1.NLL0
        
        e = B[j]

        X=toy1.set_coef([A[i].numpy() for i in range(len(A))] , Coeff0 , fix) 
        X_bis=toy1.set_coef([A_bis[i].numpy() for i in range(len(A_bis))] , Coeff0 , fix) 

        X[j]=Coeff0[j]
        X_bis[j]=Coeff0[j]

        X[j] -= np.ceil(n/2)*e
        X_bis[j] -= np.ceil(n/2)*e


        print(np.array(X)-np.array(X_bis))

        for i in range(n):
            X[j] += e 
            X_bis[j] += e

            m_100 , coef_100 = toy1.minuitfit(Ncall=100 ,verbose=False , coefini=X , fixed=fix)
            nll100=toy1.NLL
            NLL_profile100.append(nll100.numpy())
            
            m_200 , coef_200 = toy1.minuitfit(Ncall=200 ,verbose=False , coefini=X , fixed=fix)
            nll200=toy1.NLL
            NLL_profile200.append(nll200.numpy())

            m_300 , coef_300 = toy1.minuitfit(Ncall=500 ,verbose=False , coefini=X , fixed=fix)
            nll300=toy1.NLL           
            NLL_profile300.append(nll300.numpy())

            m_400 , coef_400 = toy1.minuitfit(Ncall=1000 ,verbose=False , coefini=X , fixed=fix)
            nll400=toy1.NLL
            NLL_profile400.append(nll400.numpy())

            m_500 , coef_500 = toy1.minuitfit(Ncall=10000 ,verbose=False , coefini=X , fixed=fix)
            nll500=toy1.NLL
            NLL_profile500.append(nll500.numpy())


            m , coef = toy1.minuitfit(Ncall=200 , verbose=False , coefini=X_bis , fixed=fix)
            nll=toy1.NLL
            NLL_profile.append(nll.numpy())

            '''
            m_1000 , coef_1000 = toy1.minuitfit(Ncall=1000 , verbose=False , coefini=X , fixed=fix)
            nll1000=toy1.NLL

            
            m_10000 , coef_10000 = toy1.minuitfit(Ncall=10000 , verbose=False , coefini=X , fixed=fix)
            nll10000=toy1.NLL
            '''

            print(X[j] , ' : ' , nll100.numpy()/N ,  '&' , nll200.numpy()/N ,  '&' , nll300.numpy()/N ,  '&' , nll400.numpy()/N ,  '&' , nll500.numpy()/N , '&' , nll.numpy()/N)



            x.append(X[j])

        fig, ax = plt.subplots()

        NLL_profile100=np.array(NLL_profile100)
        NLL_profile200=np.array(NLL_profile200)
        NLL_profile300=np.array(NLL_profile300)
        NLL_profile400=np.array(NLL_profile400)
        NLL_profile500=np.array(NLL_profile500)
        NLL_profile=np.array(NLL_profile)
        #NLL_profile1000=np.array(NLL_profile1000)
        #NLL_profile10000=np.array(NLL_profile1000)

        #y100=NLL_profile100-min(NLL_profile100)
        y100=NLL_profile100/N
        y200=NLL_profile200/N
        y300=NLL_profile300/N
        y400=NLL_profile400/N
        y500=NLL_profile500/N
        y=NLL_profile/N
        #y1000=NLL_profile1000-min(NLL_profile1000)
        #y1000=NLL_profile1000/10000
        #y10000=NLL_profile10000-min(NLL_profile10000)
        #y10000=NLL_profile10000/10000

        f100 = interpolate.interp1d(x, y100)
        f200 = interpolate.interp1d(x, y200)
        f300 = interpolate.interp1d(x, y300)
        f400 = interpolate.interp1d(x, y400)
        f500 = interpolate.interp1d(x, y500)
        f = interpolate.interp1d(x, y)

        #f1000 = interpolate.interp1d(x, y1000)
        #f10000 = interpolate.interp1d(x, y10000)

        X=np.linspace(x[0] , x[-1])
        #plt.plot(X , f100(X) , 'r-.' , label='100')
        #plt.plot(X , f200(X) , 'm-.' , label='200')
        plt.plot(X , f300(X) , 'b-.' , label='500')
        plt.plot(X , f400(X) , 'r-.', label='1000')
        plt.plot(X , f500(X) , 'g-.', label='5000')
        #plt.plot(X , f(X) , 'm-.', label='prof.')
       
        #plt.plot(X , f1000(X) , 'b-.', label='1000')
        #plt.plot(X , f1000(X) , 'c-.', label='1000')


        #plt.plot( x , y100  , 'k.' )
        #plt.plot( x , y200  , 'k.' )
        plt.plot( x , y300  , 'k.' )
        plt.plot( x , y400  , 'k.' )
        plt.plot( x , y500  , 'k.' )
        #plt.plot( x , y1000  , 'k.' )


       # plt.plot(x[int(np.floor(11/2))] , y100[int(np.floor(11/2))] , 'cD'  , label='MC value')
        #plt.plot(x[int(np.floor(11/2))] , y1000[int(np.floor(11/2))] , 'cD' )
        plt.plot(x[int(np.floor(11/2))] , y500[int(np.floor(11/2))] , 'cD' )
        #plt.plot(x[int(np.floor(11/2))] , y200[int(np.floor(11/2))] , 'cD' )
        plt.plot(x[int(np.floor(11/2))] , y400[int(np.floor(11/2))] , 'cD' )
        plt.plot(x[int(np.floor(11/2))] , y300[int(np.floor(11/2))] , 'cD' )

        #plt.plot(x[np.argmin(y100)] , min(y100) , 'ro' )
        #plt.plot(x[np.argmin(y200)] , min(y200) , 'mo' )
        plt.plot(x[np.argmin(y300)] , min(y300) , 'bo' )
        plt.plot(x[np.argmin(y400)] , min(y400) , 'ro' )
        plt.plot(x[np.argmin(y500)] , min(y500) , 'go'  )
        #plt.plot(x[np.argmin(y1000)] , min(y1000) , 'bo'  )


        title = ax.set_title('Expected Value:'+str(format( Coeff0[j], '.3f'))+ '      Migrad500:' +str(format( x[np.argmin(y300)], '.3f')) + '\n'+'      Migrad1000:' +str(format( x[np.argmin(y400)], '.3f')) + '      Migrad10000:' +str(format( x[np.argmin(y500)], '.3f')) )
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
