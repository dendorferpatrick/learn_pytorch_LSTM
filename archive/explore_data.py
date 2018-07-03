
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('data/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
print(dataset.head())
plt.plot(dataset)
plt.xlabel("Time")
plt.ylabel("Number of passengers")
plt.show()
