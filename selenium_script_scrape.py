from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import re
import time

browser = webdriver.Firefox()
temperatures = []
YEAR = int(input())
years = [YEAR]
for year in years:
	year_temperatures = []
	for month in range(1,13):
		browser.get("https://www.timeanddate.com/weather/india/delhi/historic?month="+str(month)+"&year="+str(year))
		date = browser.find_element_by_xpath('//*[@id="wt-his-select"]')
		options = date.find_elements_by_tag_name('option')
		timeout = 10
		for i in options:
			print(i.text)
			i.click()
			day_temperature = []
			length =  len(browser.find_element_by_xpath('//*[@id="wt-his"]/tbody').find_elements_by_tag_name('tr'))
			for i in range(1,length+1):
				try:
					ele_wait = EC.presence_of_element_located((By.XPATH, '//*[@id="wt-his"]/tbody/tr['+str(i)+']/td[2]'))
					WebDriverWait(browser, timeout).until(ele_wait)
					try:
						ele = browser.find_element_by_xpath('//*[@id="wt-his"]/tbody/tr['+str(i)+']/td[2]')
						day_temperature.append(int(re.search("[0-9][0-9]",ele.text).group(0)))
					except:
						pass
				except TimeoutException:
					pass
			print(sum(day_temperature))
			print(len(day_temperature))
			if(sum(day_temperature) == 0 or len(day_temperature) == 0):
				temperatures.append(0)
			else:
				temperatures.append(sum(day_temperature)/len(day_temperature))

	print("LENGTH = ")
	print(len(temperatures))
	for i in temperatures:
		print(i)