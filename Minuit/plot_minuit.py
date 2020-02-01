import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import b_meson_fit as bmf 
from test_iminuit import amplitude_latex_names ,amplitude_names, LaTex_labels , Standard_labels , fix_array



print(fix_array)

signal_coeffs = bmf.coeffs.signal(bmf.coeffs.SM)
Coef0=[i.numpy() for i in signal_coeffs] 
print(Coef0)


dataP = pd.read_csv("./Minuit/Test_stats/data_Pierre.csv")
dataP = dataP.values

dataI = pd.read_csv("./Minuit/Test_stats/data.csv")
dataI = dataI.values


data=np.vstack((dataP , dataI))

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

'''
def get_errors(coeffs , errors ): 
    errors = []
    mean = np.mean(coeffs)

'''

def plot_stats(data , N , save_path=None , save=False):
    LaTex=LaTex_labels(amplitude_latex_names)
    Title=Standard_labels(amplitude_names)
    coefs , errs = get_arrays(data , N)
    meanLiam=[-3.99891 , -0.16856 , +6.29739 , +0.00086 , -0.00149  , +0.53478 , +0.33127 , -0.10475 , +7.65988 , +0.28405 , 0.03372 , -0.57944 , +3.93645 , +0.11211 , -8.42401 , -0.30081 ,  +0.03393 , +0.24708 , -0.85217 , +0.14706 , -7.14570 , +7.72792 , -0.25409 , +10.48869 , +1.07207 , +1.13120 , +1.11802 , +0.99124 ]
    errsLiam=[0.06524 , 0.00809 , 0.10144 , 0.05081 , 0.00686 , 0.07358 , 0.07850 , 0.01205 , 0.10793 , 0.13308 , 0.01694 , 0.18585 , 0.04978 , 0.00618 , 0.08420 , 0.07814 , 0.00998 , 0.11652 , 0.07631 , 0.01185 , 0.10741 , 0.06190 , 0.00450 , 0.09043 , 0.01003 , 0.02296 , 0.02304 , 0.02580 ]
    j=0
    for J in range(len(coefs)):
        if fix_array[J]==0:
            A=coefs[:,J]

            fig , ax =plt.subplots()

            ave_val=np.mean(A)
            width=np.std(A)
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


            n , bins , _ =plt.hist(A  , bins = 60 , normed= True)
            print(n)
            ymin, ymax = ax.get_ylim()
            
            ax.vlines(true_val, ymin , ymax/2 , label='MC value')
            ax.set_xlim([ave_val-6*width, ave_val+6*width])
            ax.vlines(ave_val , ymin , ymax/2 , label='Average fitted value' , color='r')  
            ax.set(xlabel=LaTex[J], ylabel=r'p')
            ax.set_title('Exp Value:'+str(format(true_val, '.4f'))+' Av Value:'+str(format( ave_val, '.4f'))+'+/-'+str(format(width/np.sqrt(1000), '.4f')))
            
            #print(coefs[:,j])
            
            
            
            #xmin= meanLiam[0] - 10*np.sqrt(2400)*errsLiam[0] 
            #xmax= meanLiam[0] + 10*np.sqrt(2400)*errsLiam[0] 
            xmin , xmax = ax.get_xlim()
            X=np.linspace(xmin , xmax)
            plt.plot(X , max(n)*gaussian( X , meanLiam[j] , np.sqrt(1000)*errsLiam[j]) , '-')
            
            Xmin , Xmax = ave_val-width ,  ave_val+width/np.sqrt(1000)
            ax.axvspan(Xmin, Xmax, alpha=0.1, color='red' , label= r'$\pm \sigma$')
            ax.legend()
            j +=1
            if save:
                plt.savefig(save_path+Title[J]+'.png')







path='./Minuit/Plot_hist/'

plot_stats(data , 1000 , save_path=path , save=True )