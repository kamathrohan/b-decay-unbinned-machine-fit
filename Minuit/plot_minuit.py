import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("./Minuit/Test_stats/data.csv")
data = data.values
plt.hist(data[:0],bins = 30)
plt.show()