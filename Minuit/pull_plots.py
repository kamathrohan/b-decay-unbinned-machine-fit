import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import b_meson_fit as bmf 



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

data = np.vstack([dataP.values,dataI.values])

values, errors = get_arrays(data, 2000)

arr2d = []
for j in range(2000):
    arr = []
    for i in range(48):
        arr.append(pull(values[j][i],Coef0[i],errors[j][i]))
    arr2d.append(arr)

pullarray = np.asarray(arr2d)
array = pullarray[:,0]
array = np.sort(array)

plt.hist(array[500:1500])
plt.show()


