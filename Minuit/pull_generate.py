from toy_minuit import *
from test_iminuit import array_out_long
import csv

iterations = 48

for i in tqdm(range(iterations)):
    toy1 = toy(model = "SM")
    toy1.generate(events = 10000)
    m, coeffs = toy1.minuitfit(Ncall = 1000, coefini= toy1.get_coeffs())
    OUT = array_out_long(m)
    with open(r"./Minuit/Test_stats/data_new.csv", 'a') as data:
            writer =csv.writer(data)
            writer.writerow(OUT)
    data.close()
