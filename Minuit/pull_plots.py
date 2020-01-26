import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import b_meson_fit as bmf 
from test_iminuit import amplitude_latex_names ,amplitude_names, LaTex_labels , Standard_labels
from toy_minuit import FIX

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

dataP = pd.read_csv("./Minuit/Test_stats/data_Pierre.csv")
dataI = pd.read_csv("./Minuit/Test_stats/data.csv")
data = np.vstack((dataP.values,dataI.values))
values, errors = get_arrays(data, 2000)

pulls=(values-Coef0)/errors
for j in range(48) : 
    if FIX[j]==0:
        pulls0=np.sort(pulls[:,j])
        plt.hist(pulls0[20 : 1980] , bins=60 , color='r' , alpha= 0.6 , normed=True)
        save_path = './Minuit/Plot_pulls/'
        plt.savefig(save_path+Title[j]+'.png')
    # plt.show()

'''
arr2d = []
for j in range(2000):
    arr = []
    for i in range(48):
        arr.append(pull(values[j][i],Coef0[i],errors[j][i]))
    arr2d.append(arr)


print(len(arr2d))
for i in range(48):
    if FIX[i]==0:
        pullarray = np.asarray(arr2d)
        array = pullarray[:,i]
        array = np.sort(array)


        plt.hist(array[10:1990] , bins=50 , color='r' , alpha= 0.8)
        save_path = './Minuit/Plot_pulls/'
        plt.savefig(save_path+Title[i]+'.png')


'''


