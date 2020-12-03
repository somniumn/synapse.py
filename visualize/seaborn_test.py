import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

flights = sns.load_dataset("flights")
flights.head()
may_flights = flights.query("month == 'May'")

sns.lineplot(data=may_flights, x="year", y="passengers")
plt.title('title')
plt.grid()
plt.show()

