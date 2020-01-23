import numpy as np 
import matplotlib.pyplot as plt 
import csv 
import pandas as pd
from test_iminuit import generate_SM, minuitfit, array_out , LaTex_labels , Standard_labels, array_out_long
from tqdm import tqdm 
array=[
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


LaTex=LaTex_labels(amplitude_latex_names)
Titles=Standard_labels(amplitude_names)


iterations=1


SUM=np.zeros((iterations , 48))

for i in tqdm(range(iterations)):
    coeffs, signal_events = generate_SM(fix_array = array)
    m, Coef0, Coef_OUT = minuitfit(coeffs, signal_events , array,verbose = False)
    SUM[i ,:] = Coef_OUT
    OUT=array_out(m)
    print(OUT)


"""
np.savetxt("./Minuit/Test_stats/data.csv",SUM)
"""
with open(r"./Minuit/Test_stats/data.csv", 'a') as data:
    writer =csv.writer(data)
    for i in SUM:
        writer.writerow(i) 
    data.close()
"""
for j in range(48):
    fig , ax =plt.subplots()
    ave_val=np.mean(SUM[:,j])
    true_val=Coef0[j]
    ymin, ymax = ax.get_ylim()
    ax.vlines(true_val, ymin , ymax , label='MC value')
    ax.vlines(ave_val , ymin , ymax , label='Average fitted value' , color='r')  
    ax.set(xlabel=LaTex[j], ylabel=r'p')
    ax.set_title('Expected Value:'+str(format(true_val, '.5f'))+' Average Value:'+str(format( ave_val, '.5f')))
    ax.legend()
    #print(SUM[:,j])
    plt.hist(SUM[:,j] , align='mid')
    plt.savefig('./Minuit/Test_stats/'+Titles[j]+'.png')
    plt.close()
"""