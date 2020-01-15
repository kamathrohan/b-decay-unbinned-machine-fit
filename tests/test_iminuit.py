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
fix_array=[
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
1,1,
1,1,1,
1,1,1,
]

#Define amplitude names (total number of variables is  16*3 for alpha , beta and gamma )
amplitude_latex_names = [
    r'Re($A_{\parallel}^L$)',
    r'Im($A_{\parallel}^L$)',
    r'Re($A_{\parallel}^R$)',
    r'Im($A_{\parallel}^R$)',
    r'Re($A_{\bot}^L$)',
    r'Im($A_{\bot}^L$)',
    r'Re($A_{\bot}^R$)',
    r'Im($A_{\bot}^R$)',
    r'Re($A_{0}^L$)',
    r'Im($A_{0}^L$)',
    r'Re($A_{0}^R$)',
    r'Im($A_{0}^R$)',
    r'Re($A_{00}^L$)',
    r'Im($A_{00}^L$)',
    r'Re($A_{00}^R$)',
    r'Im($A_{00}^R$)',
]

#Number of step per fit (migrad)
Ncall=1000



def nll(signal_coeffs):
    """Get the normalized negative log likelihood
    Working with the normalised version ensures we don't need to re-optimize hyper-parameters when we
    change signal event numbers.
    Returns:
        Scalar tensor
    """
    return bmfs.normalized_nll(signal_coeffs, signal_events)

def LaTex_names(amplitude_latex_names):
    LaTex_labels=[]
    param=[r'$\alpha$(' , r'$\beta$(' , r'$\gamma$(']
    for i in amplitude_latex_names:
        for j in param:
            LaTex_labels.append(j+i+')')
    return LaTex_labels


def set_coef(Coef_INIT , Coef0,  fix_array ) :
    for i in range(len(Coef_INIT)):
        if fix_array[i]==1 :
            Coef_INIT[i]=  Coef0[i]
    return Coef_INIT



def plot_fixed(m , Coef0 ,fix_array , amplitude_latex_names):
    #First find index of non fixed parameters 
    ind=[]
    for i in range(len(fix_array)):
        if fix_array[i]==0 : 
            ind.append(int(i))

    #Now ,plot nll profile for given parameters
    LaTex_labels=LaTex_names(amplitude_latex_names)
    n=int(round(np.sqrt(len(ind))))
    fig, axs = plt.subplots(n, n)
    n1 , n2 = 0 , 0 

    #Different case if only one plot or more
    if len(ind)==1 : 
        i0=ind[0]
        X , Y =m.profile(m.parameters[i0])
        axs.plot(X , Y , label='NLL')
        ymin, ymax = axs.get_ylim()
        axs.vlines(Coef0[i0] , ymin , ymax , label='init value')
        X , Y =m.profile(m.parameters[i0])
        axs.set(xlabel=LaTex_labels[i0], ylabel=r'NLL')
        Xmin , Xmax = m.values[i0]-m.errors[i0] ,  m.values[i0]+m.errors[i0] 
        axs.axvspan(Xmin, Xmax, alpha=0.1, color='red' , label= 'interval')
        axs.set_title('Fit for '+LaTex_labels[i0]+'with N='+str(Ncall))
        axs.legend()
    else : 
        for i in ind:
            X , Y =m.profile(m.parameters[i])
            ax=axs[n1 , n2]
            ax.plot(X , Y , label='NLL')
            ymin, ymax = axs[n1 , n2].get_ylim()
            print(Coef0[i])
            ax.vlines(Coef0[i] , ymin , ymax , label='init value')
            ax.set(xlabel=LaTex_labels[i], ylabel=r'NLL')
            Xmin , Xmax = m.values[i]-m.errors[i] ,  m.values[i]+m.errors[i] 
            ax.axvspan(Xmin, Xmax, alpha=0.1, color='red' , label= 'interval')
            ax.set_title('Fit for '+LaTex_labels[i]+'with N='+str(Ncall))
            ax.legend()

            n2+=1
            if n2==n:
                n2=0
                n1+=1
    
    plt.tight_layout()
    plt.show()


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
m.migrad( ncall=Ncall)
t1=time.time()
print("Time taken to fit :", t1-t0)


##   Print fit results 
Coef_OUT=[m.values[i] for i in range(len(m.values))]
print(Coef_OUT)
print('NLL for final set of coeffs:' , nll(Coef_OUT))


##   Plot non fixed parameters 
plot_fixed(m , Coef0, fix_array , amplitude_latex_names)