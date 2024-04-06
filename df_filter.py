'''
Stuff used to create dataframes
'''
import pandas as pd

# Load the CSV file
file_path = '/Users/lok/5DE/data pull/20240316_20240406_SLD_REN_FCST_N_20240405_16_14_51_v1.csv'
df = pd.read_csv(file_path)

df = df[['OPR_DT', 'OPR_HR', 'TRADING_HUB', 'RENEWABLE_TYPE', 'MW', 'MARKET_RUN_ID']]

filtered_df = df[
    (df['TRADING_HUB'] == 'SP15') & 
    (df['RENEWABLE_TYPE'] == 'Wind') & 
    (df['MARKET_RUN_ID'].isin(['DAM', 'ACTUAL']))
]

new_file_path = '/Users/lok/5DE/processed.csv'

# Saving the filtered data to a new CSV file
filtered_df.to_csv(new_file_path, index=False)

filtered_df_actual = df[
    (df['TRADING_HUB'] == 'SP15') & 
    (df['RENEWABLE_TYPE'] == 'Wind') & 
    (df['MARKET_RUN_ID']  == 'ACTUAL')
]


new_file_path_actual = '/Users/lok/5DE/processed_actual_only.csv'

# Saving the filtered data to a new CSV file
filtered_df_actual.to_csv(new_file_path_actual, index=False)

filtered_df_predicted = df[
    (df['TRADING_HUB'] == 'SP15') & 
    (df['RENEWABLE_TYPE'] == 'Wind') & 
    (df['MARKET_RUN_ID']  == 'DAM')
]


new_file_path_predicted = '/Users/lok/5DE/processed_predicted_only.csv'

# Saving the filtered data to a new CSV file
filtered_df_predicted.to_csv(new_file_path_predicted, index=False)



file_path = '/Users/lok/5DE/processed_actual_only.csv'
data = pd.read_csv(file_path)
# sorted_df = df.sort_values(by=['OPR_DT', 'OPR_HR'])

data['Datetime'] = pd.to_datetime(data['OPR_DT']) + pd.to_timedelta(data['OPR_HR'], unit='h')

# Assuming we are summing up the energy output for each hour across all trading hubs and renewable types
# Group by the new datetime column and sum the energy output
# data_grouped = data.groupby('Datetime')['MW'].sum().reset_index()
new_file_path_merged = '/Users/lok/5DE/processed_merged.csv'

# Sort the data chronologically
data.sort_values('Datetime', inplace=True)
data.to_csv(new_file_path_merged, index=False)
