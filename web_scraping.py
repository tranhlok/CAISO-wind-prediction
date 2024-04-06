'''
Script to get the data automatically
'''

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os, time
from selenium.webdriver.common.keys import Keys

download_dir = "/Users/lok/5DE"  # Replace with your desired directory
os.makedirs(download_dir, exist_ok=True)


options = webdriver.ChromeOptions()

prefs = {
    "download.default_directory": download_dir,
    "download.directory_upgrade": "true",
    "download.prompt_for_download": "false",
    "disable-popup-blocking": "true",
    "safebrowsing.enabled": False,
    "default_content_settings": "contentSettings",
    "download": "download"
}
options.add_experimental_option("prefs", prefs)
options.add_argument("--headless")
options.add_argument("--disable-notifications")
options.add_argument('--disable-gpu')
options.add_argument('--disable-software-rasterizer')
driver = webdriver.Chrome(options=options)
options.add_argument('--safebrowsing-disable-download-protection')

# Initialize selenium webdriver with the Chrome options
driver = webdriver.Chrome(options=options)

# Initialize selenium webdriver

# Navigate to the website
driver.get('http://oasis.caiso.com/mrioasis/logon.do')

# Hover over the SYSTEM DEMAND menu
action = ActionChains(driver)

system_demand_menu = WebDriverWait(driver, 10).until(
    EC.visibility_of_element_located((By.ID, 'LowerMenuItem.oasisMenu.SysDemand')))
action.move_to_element(system_demand_menu).perform()

apply_button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.ID, 'DropdownMenu.oasisMenu.SysDemand.windSolar')))
apply_button.click()
time.sleep(2)
date_from_div = WebDriverWait(driver, 20).until(
    EC.presence_of_element_located((By.ID, 'PFC_date_from')))
date_from_input = date_from_div.find_element(By.TAG_NAME, 'input')
date_from_input.clear()
date_from_input.send_keys('03/16/2024', Keys.ENTER)

time.sleep(2)

# date_to_div = WebDriverWait(driver, 20).until(
#     EC.presence_of_element_located((By.ID, 'PFC_date_to')))
# date_to_input = date_to_div.find_element(By.TAG_NAME, 'input')
# date_to_input.clear()
# date_to_input.send_keys('4/4/2024', Keys.ENTER)

calendar_icon = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.ID, 'PFC_date_to_CAL_ICON')))
calendar_icon.click()

time.sleep(5)

today_button_to = WebDriverWait(driver, 20).until(
    EC.element_to_be_clickable((By.XPATH, "(//button[@title='Today'])[last()]")))
today_button_to.click()



time.sleep(2)

download_button = WebDriverWait(driver, 20).until(
    EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Download CSV')]")))
download_button.click()

time.sleep(20)  # Adjust this delay as needed for the download to complete

# Close the browser
driver.quit()
