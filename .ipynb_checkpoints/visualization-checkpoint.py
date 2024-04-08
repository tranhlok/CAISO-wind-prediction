'''
Use to generate some general graphs 
'''

from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.graphics.tsaplots import plot_acf

import pandas as pd
import matplotlib.pyplot as plt
file_path = '/Users/lok/5DE/processed_merged.csv'
data = pd.read_csv(file_path)
# Convert the 'Date' column to datetime and set it as index
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)

daily_results = seasonal_decompose(data['MW'], model='additive', period=24)

# Plotting the daily seasonal component
daily_results.seasonal.plot(figsize=(12, 8), title='Daily Seasonal Component of MW')
plt.show()

# Perform and plot weekly seasonal decomposition (period of 168 hours)
weekly_results = seasonal_decompose(data['MW'], model='additive', period=168)

# Plotting the weekly seasonal component
weekly_results.seasonal.plot(figsize=(12, 8), title='Weekly Seasonal Component of MW')
plt.show()
# Plot ACF for MW

plot_acf(data['MW'], lags=24*7, title='Autocorrelation Function of MW')
plt.show()
