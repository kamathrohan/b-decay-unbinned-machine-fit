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
0,1,1
]

fix_alphas=[
0,1,1,
0,1,1,
0,1,1,
0,1,1,
0,1,1,
0,1,1,
0,1,1,
1,1,1,
0,1,1,
1,1,1,
1,1,1,
1,1,1,
0,1,1,
0,1,1,
0,1,1,
0,1,1]

fix_one_alpha=[
0,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
0,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1,
1,1,1]


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


def generate_SM(events = 2400, fix_array = fix_array, verbose = False):
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

'''
def nll(signal_coeffs):
    """
    Return negative of the log likelihood for given events based on particular amplitude coefficients

    Args:
        coeffs: List of scalar coefficient tensors
        events: Tensor of shape (N, 4) with axis 1 representing params [q2, cos_theta_k, cos_theta_l, phi]

    Returns:
        Scalar: negative log likelihood

    """
    return bmfs.nll(signal_coeffs, signal_events)
'''


def plot_profiles(m , Coef0 ,coeffout,fix_array , amplitude_latex_names, save_path , show = True, save = False):
    #First find index of non fixed parameters 
    ind=[]
    for i in range(len(fix_array)):
        if fix_array[i]==0 : 
            ind.append(int(i))

    #Now ,plot nll profile for given parameters
    LaTex=LaTex_labels(amplitude_latex_names)
    Title=Standard_labels(amplitude_names)    

    #Different case if only one plot or more
    if len(ind)==1 : 
        fig, ax = plt.subplots()
        i0=ind[0]
        X , Y =m.profile(m.parameters[i0])
        ax.plot(X , Y , label='NLL')
        ymin, ymax = ax.get_ylim()
        ax.vlines(Coef0[i0] , ymin , ymax , label='MC value')
        ax.vlines(m.values[i0] , ymin , ymax , label='Fitted value' , color='r')
        X , Y =m.profile(m.parameters[i0])
        ax.set(xlabel=LaTex[i0], ylabel=r'NLL')
        Xmin , Xmax = m.values[i0]-m.errors[i0] ,  m.values[i0]+m.errors[i0] 
        ax.axvspan(Xmin, Xmax, alpha=0.1, color='red' , label= r'$\pm \sigma$')
        ax.set_title('Fit for '+LaTex[i0]+'with N='+str(Ncall)+'\n Expected Value:'+str(Coef0[i0])+' Actual Value:'+str(coeffout[i0]))
        ax.legend()
        titre = Title[i0]

        #Save figures of save= True 
        if save : 
            plt.savefig(save_path+titre+'.png')
    else : 
        for i in ind:
            #Draw profile along parameter parameters[i]
            X , Y =m.profile(m.parameters[i])
            fig, ax = plt.subplots()
            ax.plot(X , Y , label='NLL')
            ymin, ymax = ax.get_ylim()
            ax.vlines(Coef0[i] , ymin , ymax , label='MC value')
            ax.vlines(m.values[i] , ymin , ymax , label='fitted value' , color='r')
            ax.set(xlabel=LaTex[i], ylabel=r'NLL')
            Xmin , Xmax = m.values[i]-m.errors[i] ,  m.values[i]+m.errors[i] 
            ax.axvspan(Xmin, Xmax, alpha=0.1, color='red' , label= r'$\pm \sigma$')
            ax.set_title('Expected Value:'+str(format(Coef0[i], '.5f'))+' Actual Value:'+str(format(coeffout[i], '.5f')))
            ax.legend()

            #Save figures of save= True 
            if save:
                plt.savefig(save_path+Title[i]+'.png')

        #Title = 'Likelihood profiles for '+str(len(ind))+' parameters'
    plt.tight_layout()

    if show:
        plt.show()
  
def minuitfit(Coef0, signal_events,fix_array, Ncall=10000, verbose = False):
    A=bmf.coeffs.fit(bmf.coeffs.fit_initialization_scheme_default)
    Coef_INIT=[A[i].numpy() for i in range(len(A))]
    Coef_INIT=set_coef(Coef_INIT, Coef0, fix_array)
    
    def nll_iminuit(signal_coeffs):
        return bmfs.nll(signal_coeffs, signal_events)

    if verbose:
        print("Coeffs used for MC:", Coef0)
        print("Initial coeffs for minuit fit:", Coef_INIT)
    m = Minuit.from_array_func(nll_iminuit,Coef_INIT, fix= fix_array , errordef=0.5, pedantic = False)
    t0=time.time()

    M=m.get_fmin()

    m.migrad( ncall=Ncall)
    M=m.get_fmin()

    print(M)
    if M.is_above_max_edm==False  :
        bol=0
    else :
        bol=1

    t1=time.time()
    if verbose:
        print("Time taken to fit :", t1-t0)
    ##   Print fit results 
    Coef_OUT=[m.values[i] for i in range(len(m.values))]
    if verbose:
        print(Coef_OUT)
        print('NLL for final set of coeffs:' , nll_iminuit(Coef_OUT))
    return m, Coef0, Coef_OUT , bol

def array_out(m):
    n=len(m.parameters)
    OUT=np.zeros((n , 2))
    for i in range(n):
        OUT[i , :] = [ m.values[i] , m.errors[i]]
    return OUT


def array_out_long(m):
    n=len(m.parameters)
    OUT = []
    for i in range(n):
        OUT.append(m.values[i])
        OUT.append(m.errors[i])
    return OUT
