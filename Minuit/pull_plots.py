import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd 
import b_meson_fit as bmf 
from test_iminuit import amplitude_latex_names ,amplitude_names, LaTex_labels , Standard_labels
from toy_minuit import FIX
from scipy.stats import norm


LaTex=LaTex_labels(amplitude_latex_names)
Title=Standard_labels(amplitude_names)


def pull(value, expected, sigma):
    return (value-expected)/sigma

def get_arrays(data , N):
    '''
        Inputs : data (2d array format) , N (number of migrad fits)
        
        Splits the csv data in two arrays for amplitudes and associated errors, both of dimension (N , 48) 
    '''
    raw_data=data[0:N,:]
    coefs , errors = np.zeros((N , 48)) , np.zeros((N , 48))
    for i in range(0,96,2):
        coefs[: , int(i/2)]= raw_data[: , i]
        errors[: , int(i/2)]= raw_data[: , i+1]
    return coefs , errors

signal_coeffs = bmf.coeffs.signal(bmf.coeffs.SM)
Coef0=[i.numpy() for i in signal_coeffs] 

data = pd.read_csv("./Minuit/Test_stats/data_new.csv").values
#data = np.vstack((dataP.values,dataI.values))
values, errors = get_arrays(data, 1000)

pulls=(values-Coef0)/errors
for j in range(48) : 
    if FIX[j]==0 :
        _ , ax = plt.subplots()
        pulls0=np.sort(pulls[:,j])

        #plt.hist(pulls0[20 : 1980] , bins=60 , color='r' , alpha= 0.3 , normed=True)
        ymin , ymax = ax.get_ylim()
        ax.vlines( 0 , ymin , ymax , label='0' , color='k' , linestyles='dashed')
        av0=np.mean(pulls0[20 : 1980])
        std0=np.std(pulls0[20 : 1980])
        ax.vlines( av0 , ymin , ymax , label='average'  , color='b' , linestyles='dashed')
        ax.axvspan(av0-std0, av0+std0, alpha=0.1, color='b' , label= r'$\pm \sigma$')
        ax.legend()
        ax.set_title('Av value:'+str(format( av0, '.3f'))+'+/-'+str(format(std0, '.3f')))

        mu, sigma = norm.fit(pulls0)

        n, bins, patches = plt.hist(pulls0, bins=60 , color='r' , alpha= 0.6 , normed=True)
        y = mlab.normpdf(bins, mu, sigma)
        plt.plot(bins,y, 'r--')

        save_path = './Minuit/Plot_pulls/Plot_pulls_InitCurrent/'
        plt.title(LaTex[j]+'\n Mean Pull: '+str(mu)+" Sigma: "+str(sigma))
        plt.savefig(save_path+Title[j]+'.png')
        plt.close()
    # plt.show()





