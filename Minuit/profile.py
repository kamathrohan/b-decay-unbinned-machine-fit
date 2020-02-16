import numpy as np 
import matplotlib.pyplot as plt 
import b_meson_fit as bmf 
from toy_minuit import toy
import time 
from tqdm import tqdm
import csv 
from test_iminuit import amplitude_latex_names ,amplitude_names, LaTex_labels , Standard_labels  , fix_array , fix_alphas , fix_one_alpha , fix_alpha_beta
from scipy import interpolate 
from textwrap import wrap


LaTex=LaTex_labels(amplitude_latex_names)
Title=Standard_labels(amplitude_names)
init_scheme    = bmf.coeffs.fit_initialization_same #fit_initialization_scheme_default #fit_initialization_any
init_scheme_1  = bmf.coeffs.fit_initialization_scheme_default
FIX=fix_alpha_beta #fix_array #fix_alphas #fix_one_alpha #fix_alpha_beta
N=10000

toy1 = toy( model='SM')

toy1.generate( events = N , verbose = False )
A0=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_default , current_signal_model=toy1.model)
Coeff0=[A0[i].numpy() for i in range(len(A0))]
m , coef = toy1.minuitfit(Ncall=10000 , verbose=False , coefini=Coeff0 , fixed=FIX)


B=[m.errors[i] for i in m.errors]
B=np.array(B)


path='./Minuit/Plot_profiles/TryIt/'
n=11
J=0
idx=48
Nmigrad=[100 , 200 , 500 , 1000 , 10000]
NLL=np.zeros((len(Nmigrad) , n))


A=bmf.coeffs.fit(init_scheme , current_signal_model=toy1.model)
A_bis=bmf.coeffs.fit(init_scheme_1 , current_signal_model=toy1.model)


for j in tqdm(range(0 , idx)):

    if FIX[j]==0:
        print(j)
        
        fix=FIX[:]
        fix[j]=1
        x=[]
        
        
        NLL_profile1=[] 
        NLL_profile2=[] 
        NLL_profile3=[] 
        NLL_profile4=[] 
        NLL_profile5=[]  

        NLL_profile1_bis=[] 
        NLL_profile2_bis=[] 
        NLL_profile3_bis=[] 
        NLL_profile4_bis=[] 
        NLL_profile5_bis=[]  


        NLL_profile10=[]
        NLL_profile100=[]
        nll0=toy1.NLL0
        
        e = B[j]

        X=toy1.set_coef([A[i].numpy() for i in range(len(A))] , Coeff0 , fix) 
        X_bis=toy1.set_coef([A_bis[i].numpy() for i in range(len(A_bis))] , Coeff0 , fix) 

        X[j]=Coeff0[j]
        X_bis[j]=Coeff0[j]

        X[j] -= np.ceil(n/2)*e
        X_bis[j] -= np.ceil(n/2)*e

        for i in range(n):
            X[j] += e 
            X_bis[j] += e

            m_1 , coef_1 = toy1.minuitfit(Ncall=100 ,verbose=False , coefini=X , fixed=fix)
            nll1=toy1.NLL
            NLL_profile1.append(nll1.numpy())
            
            m_2 , coef_2 = toy1.minuitfit(Ncall=200 ,verbose=False , coefini=X , fixed=fix)
            nll2=toy1.NLL
            NLL_profile2.append(nll2.numpy())

            m_3 , coef_3 = toy1.minuitfit(Ncall=500 ,verbose=False , coefini=X , fixed=fix)
            nll3=toy1.NLL           
            NLL_profile3.append(nll3.numpy())

            m_4 , coef_4 = toy1.minuitfit(Ncall=1000 ,verbose=False , coefini=X , fixed=fix)
            nll4=toy1.NLL
            NLL_profile4.append(nll4.numpy())

            m_5 , coef_5 = toy1.minuitfit(Ncall=10000 ,verbose=False , coefini=X , fixed=fix)
            nll5=toy1.NLL
            NLL_profile5.append(nll5.numpy())


            m_1_bis , coef_1_bis = toy1.minuitfit(Ncall=100 , verbose=False , coefini=X_bis , fixed=fix)
            nll1_bis=toy1.NLL
            NLL_profile1_bis.append(nll1_bis.numpy())

            m_2_bis , coef_3_bis = toy1.minuitfit(Ncall=200 , verbose=False , coefini=X_bis , fixed=fix)
            nll2_bis=toy1.NLL
            NLL_profile2_bis.append(nll2_bis.numpy())

            m_3_bis , coef_3_bis = toy1.minuitfit(Ncall=500 , verbose=False , coefini=X_bis , fixed=fix)
            nll3_bis=toy1.NLL
            NLL_profile3_bis.append(nll3_bis.numpy())

            m_4_bis , coef_4_bis = toy1.minuitfit(Ncall=1000 , verbose=False , coefini=X_bis , fixed=fix)
            nll4_bis=toy1.NLL
            NLL_profile4_bis.append(nll4_bis.numpy())

            m_5_bis , coef_5_bis = toy1.minuitfit(Ncall=10000 , verbose=False , coefini=X_bis , fixed=fix)
            nll5_bis=toy1.NLL
            NLL_profile5_bis.append(nll5_bis.numpy())


            '''
            m_10 , coef_1000 = toy1.minuitfit(Ncall=1000 , verbose=False , coefini=X , fixed=fix)
            nll10=toy1.NLL

            
            m_1 , coef_10000 = toy1.minuitfit(Ncall=10000 , verbose=False , coefini=X , fixed=fix)
            nll100=toy1.NLL
            '''

            print(X[j] , ' : ' , nll1.numpy()/N ,  '&' , nll2.numpy()/N ,  '&' , nll3.numpy()/N ,  '&' , nll4.numpy()/N ,  '&' , nll5.numpy()/N , '&' ,'\n',  nll1_bis.numpy()/N, '&' , nll2_bis.numpy()/N, '&' , nll3_bis.numpy()/N, '&' , nll4_bis.numpy()/N, '&' , nll5_bis.numpy()/N , '\n')

            x.append(X[j])



        fig, ax = plt.subplots()

        NLL_profile1=np.array(NLL_profile1)
        NLL_profile2=np.array(NLL_profile2)
        NLL_profile3=np.array(NLL_profile3)
        NLL_profile4=np.array(NLL_profile4)
        NLL_profile5=np.array(NLL_profile5)
        NLL_profile1_bis=np.array(NLL_profile1_bis)
        NLL_profile2_bis=np.array(NLL_profile2_bis)
        NLL_profile3_bis=np.array(NLL_profile3_bis)
        NLL_profile4_bis=np.array(NLL_profile4_bis)
        NLL_profile5_bis=np.array(NLL_profile5_bis)

        #NLL_profile10=np.array(NLL_profile10)
        #NLL_profile100=np.array(NLL_profile10)

        #y1=NLL_profile1-min(NLL_profile1)
        y1=NLL_profile1/N
        y2=NLL_profile2/N
        y3=NLL_profile3/N
        y4=NLL_profile4/N
        y5=NLL_profile5/N
        y1_bis=NLL_profile1_bis/N
        y2_bis=NLL_profile2_bis/N
        y3_bis=NLL_profile3_bis/N
        y4_bis=NLL_profile4_bis/N
        y5_bis=NLL_profile5_bis/N
        #y1000=NLL_profile10-min(NLL_profile10)
        #y1000=NLL_profile10/10000
        #y10000=NLL_profile100-min(NLL_profile100)
        #y10000=NLL_profile100/10000

        f1 = interpolate.interp1d(x, y1)
        f2 = interpolate.interp1d(x, y2)
        f3 = interpolate.interp1d(x, y3)
        f4 = interpolate.interp1d(x, y4)
        f5 = interpolate.interp1d(x, y5)
        f1_bis = interpolate.interp1d(x, y1_bis)
        f2_bis = interpolate.interp1d(x, y2_bis)
        f3_bis = interpolate.interp1d(x, y3_bis)
        f4_bis = interpolate.interp1d(x, y4_bis)
        f5_bis = interpolate.interp1d(x, y5_bis)


        #f1000 = interpolate.interp1d(x, y1000)
        #f10000 = interpolate.interp1d(x, y10000)

        X=np.linspace(x[0] , x[-1])
        #plt.plot(X , f1(X) , '-.' , color='red' , label='rand100')
        #plt.plot(X , f2(X) , '-.' , color='darkorange' , label='rand200')
        #plt.plot(X , f3(X) , '-.' , color='orange' , label='rand500')
        plt.plot(X , f4(X) , '-.' , color='yellow' , label='rand1000')
        plt.plot(X , f5(X) , '-.' , color='yellowgreen' , label='rand10000')

        #plt.plot(X , f1_bis(X) , '-.' , color='darkblue' , label='par100')
        #plt.plot(X , f2_bis(X) , '-.' , color='blue' , label='par200')
        #plt.plot(X , f3_bis(X) , '-.' , color='cyan' , label='par500')
        plt.plot(X , f4_bis(X) , '-.' , color='magenta' , label='par1000')
        plt.plot(X , f5_bis(X) , '-.' , color='pink' , label='par10000')
       
        #plt.plot(X , f1000(X) , 'b-.', label='1000')
        #plt.plot(X , f1000(X) , 'c-.', label='1000')


        #plt.plot( x , y1  , 'k.' )
        #plt.plot( x , y2  , 'k.' )
        #plt.plot( x , y3  , 'k.' )
        plt.plot( x , y4  , 'k.' )
        plt.plot( x , y5  , 'k.' )

        #plt.plot( x , y1_bis  , 'k.' )
        #plt.plot( x , y2_bis  , 'k.' )
        #plt.plot( x , y3_bis  , 'k.' )
        plt.plot( x , y4_bis  , 'k.' )
        plt.plot( x , y5_bis  , 'k.' )


       # plt.plot(x[int(np.floor(11/2))] , y1[int(np.floor(11/2))] , 'cD'  , label='MC value')
        #plt.plot(x[int(np.floor(11/2))] , y1[int(np.floor(11/2))] , 'cD' )
        #plt.plot(x[int(np.floor(11/2))] , y5[int(np.floor(11/2))] , 'cD' )
        #plt.plot(x[int(np.floor(11/2))] , y2[int(np.floor(11/2))] , 'cD' )
        #plt.plot(x[int(np.floor(11/2))] , y4[int(np.floor(11/2))] , 'cD' )
        #plt.plot(x[int(np.floor(11/2))] , y3[int(np.floor(11/2))] , 'cD' )
        ymin, ymax = ax.get_ylim()
        ax.vlines(x[int(np.floor(11/2))] , min(y5_bis) , min(y5_bis)+0.0005)
        ax.fill_between(X, f5_bis(X), min(y5_bis)+0.00005 , where= min(y5_bis)+0.00005 > f5_bis(X) , alpha=0.4, color='red' , label=r' $\sigma$')
        ax.fill_between(X, f5_bis(X), min(y5_bis)+0.0001 , where= min(y5_bis)+0.0001 > f5_bis(X) , alpha=0.2, color='orange' , label=r' 2$\sigma$')
        ax.fill_between(X, f5_bis(X), min(y5_bis)+0.00015 , where= min(y5_bis)+0.00015 > f5_bis(X) , alpha=0.1, color='yellow' , label=r' 3$\sigma$')

        #plt.plot(x[np.argmin(y1)] , min(y1) , 'r.' )
        #plt.plot(x[np.argmin(y2)] , min(y2) , 'r.' )
        #plt.plot(x[np.argmin(y3)] , min(y3) , 'r.' )
        plt.plot(x[np.argmin(y4)] , min(y4) , 'r.' )
        plt.plot(x[np.argmin(y5)] , min(y5) , 'r.' )

        #plt.plot(x[np.argmin(y1_bis)] , min(y1_bis) , 'r.' )
        #plt.plot(x[np.argmin(y2_bis)] , min(y2_bis) , 'r.' )
        #plt.plot(x[np.argmin(y3_bis)] , min(y3_bis) , 'r.' )
        plt.plot(x[np.argmin(y4_bis)] , min(y4_bis) , 'r.' )
        plt.plot(x[np.argmin(y5_bis)] , min(y5_bis) , 'r.' )

        """
        plt.plot(x[np.argmin(y4)] , min(y4) , 'ro' )
        plt.plot(x[np.argmin(y5)] , min(y5) , 'go'  )
        plt.plot(x[np.argmin(y1_bis)] , min(y1_bis) , 'bo'  )
        """

        title = ax.set_title('Expected Value:'+str(format( Coeff0[j], '.3f'))+ '      Migrad500:' +str(format( x[np.argmin(y3)], '.3f')) + '\n'+'      Migrad1000:' +str(format( x[np.argmin(y4)], '.3f')) + '      Migrad10000:' +str(format( x[np.argmin(y5)], '.3f')) )
        ax.set_ylim([min(y5_bis)-0.00001 , min(y5_bis)+0.001])
        #+r'$\pm$'+str(format(B[j], '.4f')) +
        #ax.vlines( toy1.coeffs[j]  , ymin , ymax , label='MC value')
        #plt.show()
        
        #ax.fill_between(X, f1(X), 0.5 , where=0.5 > f1(X) , alpha=0.5, color='red') 

        ax.legend()
        ax.set(xlabel=LaTex[j], ylabel=r' NLL')        #ax.axhspan(ymin, 0.5, alpha=0.1, color='red' , label= r'$\pm \sigma$')        
        plt.subplots_adjust(top=0.9)
        plt.savefig(path+Title[j]+'.png')
