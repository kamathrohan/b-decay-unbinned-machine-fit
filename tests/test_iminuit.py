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
#Define names #


#Define boolean sequence of variables to keep fixed 
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

amplitude_names = [
    'a_para_l_re',
    'a_para_l_im',
    'a_para_r_re',
    'a_para_r_im',
    'a_perp_l_re',
    'a_perp_l_im',
    'a_perp_r_re',
    'a_perp_r_im',
    'a_0_l_re',
    'a_0_l_im',
    'a_0_r_re',
    'a_0_r_im',
    'a_00_l_re',
    'a_00_l_im',
    'a_00_r_re',
    'a_00_r_im',
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

def LaTex_labels(amplitude_latex_names):
    LaTex_labels=[]
    param=[r'$\alpha$(' , r'$\beta$(' , r'$\gamma$(']
    for i in amplitude_latex_names:
        for j in param:
            LaTex_labels.append(j+i+')')
    return LaTex_labels


def Standard_labels(amplitude_names):
    labels=[]
    param=['alpha(' , 'beta(' , 'gamma(']
    for i in amplitude_names:
        for j in param:
            labels.append(j+i+')')
    return labels

def set_coef(Coef_INIT , Coef0,  fix_array ) :
    for i in range(len(Coef_INIT)):
        if fix_array[i]==1 :
            Coef_INIT[i]=  Coef0[i]
    return Coef_INIT

def plot_fixed(m , Coef0 ,fix_array , amplitude_latex_names, coeffout,show = True, save = False):
    #First find index of non fixed parameters 
    ind=[]
    for i in range(len(fix_array)):
        if fix_array[i]==0 : 
            ind.append(int(i))

    #Now ,plot nll profile for given parameters
    LaTex=LaTex_labels(amplitude_latex_names)
    Title=Standard_labels(amplitude_names)
    n=int(math.ceil(np.sqrt(len(ind))))
    fig, axs = plt.subplots(n, n)
    n1 , n2 = 0 , 0 

    #Different case if only one plot or more
    if len(ind)==1 : 
        i0=ind[0]
        X , Y =m.profile(m.parameters[i0])
        axs.plot(X , Y , label='NLL')
        ymin, ymax = axs.get_ylim()
        axs.vlines(Coef0[i0] , ymin , ymax , label='init value')
        axs.vlines(m.values[i0] , ymin , ymax , label='fitted value' , color='r')
        X , Y =m.profile(m.parameters[i0])
        axs.set(xlabel=LaTex[i0], ylabel=r'NLL')
        Xmin , Xmax = m.values[i0]-m.errors[i0] ,  m.values[i0]+m.errors[i0] 
        axs.axvspan(Xmin, Xmax, alpha=0.1, color='red' , label= r'$\pm \sigma$')
        axs.set_title('Fit for '+LaTex[i0]+'with N='+str(Ncall)+'\n Expected Value:'+str(Coef0[i0])+' Actual Value:'+str(coeffout[i0]))
        axs.legend()
        titre = Title[i0]
    else : 
        for i in ind:
            X , Y =m.profile(m.parameters[i])
            #ax=axs[n1 , n2]
            fig, ax = plt.subplots(1, 1)
            ax.plot(X , Y , label='NLL')
            ymin, ymax = ax.get_ylim()
            ax.vlines(Coef0[i] , ymin , ymax , label='init value')
            ax.vlines(m.values[i] , ymin , ymax , label='fitted value' , color='r')
            ax.set(xlabel=LaTex[i], ylabel=r'NLL')
            Xmin , Xmax = m.values[i]-m.errors[i] ,  m.values[i]+m.errors[i] 
            ax.axvspan(Xmin, Xmax, alpha=0.1, color='red' , label= r'$\pm \sigma$')
            #ax.set_title(LaTex[i])
            ax.set_title('Expected Value:'+str(format(Coef0[i], '.5f'))+' Actual Value:'+str(format(coeffout[i], '.5f')))
            ax.legend()
            plt.savefig('./Minuit/N=1000000/'+Title[i]+'.png')
            n2+=1
            if n2==n:
                n2=0
                n1+=1
        #Title = 'Likelihood profiles for '+str(len(ind))+' parameters'
    plt.tight_layout()
    
    if save:
        plt.savefig('./Minuit/Test/'+titre+'.png')
    if show:
        plt.show()
    
def generate(events = 24000, fix_array = fix_array, verbose = False):
    """
    Generate Data for fitting
    Returns Amplitude Coefficients, and events 
    """

    signal_coeffs = bmf.coeffs.signal(bmf.coeffs.SM)
    Coef0=[i.numpy() for i in signal_coeffs] 
    if verbose:  
        print("Ideal coeffs for SM:", Coef0 )
    t0=time.time()
    signal_events = bmf.signal.generate(signal_coeffs, events_total=events)
    t1=time.time()
    if verbose:
        print("Time taken to generate data:", t1-t0)
    return  Coef0, signal_events


def minuitfit(Coef0, signal_events,fix_array, Ncall=1000,verbose = False):
    A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_default)
    Coef_INIT=[A[i].numpy() for i in range(len(A))]
    Coef_INIT=set_coef(Coef_INIT, Coef0, fix_array)
    if verbose:
        print("Initial coeffs for minuit fit:", Coef_INIT)
    m = Minuit.from_array_func(nll,Coef_INIT, fix= fix_array , errordef=0.5, pedantic = False)
    t0=time.time()
    m.migrad( ncall=Ncall)
    t1=time.time()
    if verbose:
        print("Time taken to fit :", t1-t0)
    ##   Print fit results 
    Coef_OUT=[m.values[i] for i in range(len(m.values))]
    if verbose:
        print(Coef_OUT)
        print('NLL for final set of coeffs:' , nll(Coef_OUT))
    return m, Coef0, Coef_OUT




'''
for i in tqdm(range(48)):
    array =[
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
        1,1,1,
        1,1,1,
        ]
    array[i] = 0
    coeffs, signal_events = generate(fix_array = array)
    m , Coef0, Coef_OUT = minuitfit(coeffs,signal_events,fix_array = array, Ncall = 1000)
    plot_fixed(m , Coef0, array , amplitude_latex_names,Coef_OUT,show = False, save=True)
'''

array =[
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


coeffs, signal_events = generate(fix_array = array)
m , Coef0, Coef_OUT = minuitfit(coeffs,signal_events,fix_array = array, Ncall = 10000 , verbose=True)
A=m.hesse()
print(A)