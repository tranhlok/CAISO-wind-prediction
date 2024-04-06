from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import datetime
import pandas as pd
import time


def download_wind_generation_data():
    # Calculate the required date range
    start_date = (datetime.datetime.now() - datetime.timedelta(days=22)).strftime('%Y-%m-%d')
    end_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    # Initialize selenium webdriver
    driver = webdriver.Chrome()  # or the appropriate driver for your browser
    wait = WebDriverWait(driver, 10)

    # Navigate to the website and login if necessary
    driver.get('http://oasis.caiso.com/mrioasis/logon.do')

    system_demand_menu = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.ID, 'oasisMenu.oasisMenuSysDemand')))
    system_demand_menu.click()

# Wait for the submenu item to be visible and click it
# The ID for the submenu item would need to be retrieved in a similar way you've done for the menu
# Example: 'oasisMenu.oasisMenuSysDemand.windSolar'

# Here we use a time sleep to wait for the submenu to appear, since we do not have its exact ID or mechanism
# In practice, you would wait for the specific element to be clickable or visible using WebDriverWait
    time.sleep(3)  # Replace with explicit wait for the submenu item

# If the submenu item is a link you could navigate directly by finding the link.
# If it's a button or an item that needs to be clicked, you would do the following:
    wind_solar_forecast = driver.find_element_by_id('GC_197786213')  # Replace with the actual submenu ID
    wind_solar_forecast.click()

# From here, you would continue with date input, applying filters, and downloading CSV file as needed
# Rest of your scraping logic

# Don't forget to close the browser after you're done
    driver.quit()


def filter_csv_data(csv_file_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Filter based on the requirements
    filtered_df = df[(df['TRADING_HUB'] == 'SP15') & (df['RENEWABLE_TYPE'] == 'Wind')]
    filtered_df = filtered_df[filtered_df['MARKET_RUN_ID'].isin(['DAM', 'ACTUAL'])]

    # Return the filtered dataframe
    return filtered_df

# Main execution
if __name__ == '__main__':
    download_wind_generation_data()
    # Assuming the downloaded file path is known
    filtered_data = filter_csv_data('downloaded_file.csv')
    # Further processing or saving the filtered data
