import numpy as np 
import matplotlib.pyplot as plt 
import csv 
import pandas as pd
from test_iminuit import generate_SM, minuitfit, array_out , LaTex_labels , Standard_labels, array_out_long
from tqdm import tqdm 


iterations=1000


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
for i in tqdm(range(iterations)):
    coeffs, signal_events = generate_SM(fix_array = array)
    m, Coef0, Coef_OUT , bol= minuitfit(coeffs, signal_events , array,verbose = False)
    OUT=array_out_long(m)
    if bol==1 : 
        with open(r"./Minuit/Test_stats/data_Pierre.csv", 'a') as data:
            writer =csv.writer(data)
            writer.writerow(OUT) 
        data.close()