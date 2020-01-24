import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import b_meson_fit as bmf 
from test_iminuit import amplitude_latex_names ,amplitude_names, LaTex_labels , Standard_labels





signal_coeffs = bmf.coeffs.signal(bmf.coeffs.SM)
Coef0=[i.numpy() for i in signal_coeffs] 
print(Coef0)


dataP = pd.read_csv("./Minuit/Test_stats/data_Pierre.csv")
dataP = dataP.values

dataI = pd.read_csv("./Minuit/Test_stats/data.csv")
dataI = dataI.values


data=np.vstack((dataP , dataI))
print(type(data))

def get_arrays(data , N):
    '''
        Inputs : data (2d array format) , N (number of migrad fits)
        
        Splits the csv data in two arrays for amplitudes and associated errors, both of dimension (N , 48) 
    '''
    raw_data=data[0:N,:]
    print('Shape input array :' , np.shape(raw_data))
    coefs , errors = np.zeros((N , 48)) , np.zeros((N , 48))
    print('Shape output arrays :' , np.shape(coefs) , np.shape(errors))
    for i in range(0,96,2):
        coefs[: , int(i/2)]= raw_data[: , i]
        errors[: , int(i/2)]= raw_data[: , i+1]
    return coefs , errors

def gaussian(x, mu, sig):
    #(1/(np.sqrt(2*np.pi*np.power(sig , 2.))))*
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def plot_stats(data , N , save_path=None , save=False):
    LaTex=LaTex_labels(amplitude_latex_names)
    Title=Standard_labels(amplitude_names)
    coefs , errs = get_arrays(data , N)
    meanLiam=[-3.99891 , -0.16856 , +6.29739 , +0.00086 , -0.00149  , +0.53478 , +0.33127 , -0.10475 , +7.65988 , +0.28405 , 0.03372 , -0.57944 , +3.93645 , +0.11211 , -8.42401 , -0.30081 ,  +0.03393 , +0.24708 , -0.85217 , +0.14706 , -7.14570 , +7.72792 , -0.25409 , +10.48869 , +1.07207 , +1.13120 , +1.11802 , +0.99124 ]
    errsLiam=[0.06524 , 0.00809 ]
    for J in range(len(coefs)):
        
        A=coefs[:,J]

        fig , ax =plt.subplots()

        ave_val=np.mean(A)
        width=np.std(A)
        print('   ___________________________________________')
        print('   ___________________________________________')
        print(Title[J])

        print(width)
        true_val=Coef0[J]
        clean_idx=[]
        for a in range(len(A)):
            
            if abs(A[a]) > width/1 :
                #print(a)
                #print(Coef0[J] , A[a])
                clean_idx.append(a)

        ymin, ymax = ax.get_ylim()
        ax.vlines(true_val, ymin , ymax/2 , label='MC value')
        ax.set_xlim([ave_val-6*width, ave_val+6*width])
        ax.vlines(ave_val , ymin , ymax/2 , label='Average fitted value' , color='r')  
        ax.set(xlabel=LaTex[J], ylabel=r'p')
        ax.set_title('Exp Value:'+str(format(true_val, '.4f'))+' Av Value:'+str(format( ave_val, '.4f'))+'+/-'+str(format(width, '.4f')))
        
        #print(coefs[:,j])
        n , bins , _ =plt.hist(A  , bins = 80 , normed= True)
        
        
        #GAUSS=np.sort(A)
        #plt.plot(GAUSS , gaussian( GAUSS , meanLiam[0] , errsLiam[0]) , '-')
        
        Xmin , Xmax = ave_val-width ,  ave_val+width
        ax.axvspan(Xmin, Xmax, alpha=0.1, color='red' , label= r'$\pm \sigma$')
        ax.legend()
        if save:
            plt.savefig(save_path+Title[J]+'.png')







path='./Minuit/Test_stats/'


plot_stats(data , 1000 , save_path=path , save=True )