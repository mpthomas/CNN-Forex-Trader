
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import configparser
import time

class Nadex():
	def __init__(self):
		self.in_results_frame=False

		try:
			config=configparser.RawConfigParser()
			config.read('config/nadex.cfg')

			self.chrome=config.get('nadex','chrome')
			self.login_url=config.get('nadex','login_url')
			self.demo=config.get('nadex','demo')

			config=configparser.RawConfigParser()
			config.read('config/passwd.cfg')

			self.username=config.get('nadex','username')
			print ("Finished getting login")
			self.passwd=config.get('nadex','password')

		except Exception as e:
			print(e)
			return None

		#self.driver=webdriver.Chrome(self.chrome)
		self.driver=webdriver.Remote(
			command_executor='http://127.0.0.1:4444/wd/hub',
			desired_capabilities=DesiredCapabilities.CHROME)
		#self.driver.start()

	def login(self):
		self.driver.get(str(self.login_url))

		#time.sleep(5)
	
		el=self.driver.find_element_by_id("account_id")
		if(el):
			el.send_keys(self.username)
		else:
			print("login form not found. Might already be logged in")
			return False

		self.driver.find_element_by_id("password").send_keys(self.passwd)

		if(self.demo):
			self.driver.find_element_by_id("demo-toggle").click()

		self.driver.find_element_by_id("loginbutton").click()

	def get_contracts_old(self):
		self.driver.refresh()
		time.sleep(10)

		try:
			wait=WebDriverWait(self.driver,100)
			wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID,'ifrMarketFilter')))
		except Exception as e:
			print(e)
			return []

		try:	
			self.driver.find_element_by_id("ddDropDown_filterNameSelect").click()
			self.driver.find_element_by_css_selector("div[data='EUR/USD']").click()
			#self.driver.execute_script("arguments[0].setAttribute('data','EUR/USD')",e)
			self.driver.find_element_by_id("btnSearchFilter").click()
		except Exception as e:
			print(e)
			return []

		try:
			self.driver.switch_to_default_content()
			wait.WebDriverWait(self.driver,10)
			wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID,'ifrFilterResults')))
		except Exception as e:
			print("Couldn't get results iframe")
			return []

		contracts=self.driver.find_elements(By.CLASS_NAME,"contract")
		if(len(contracts) == 0):
			print("No contracts found")
			return []

		for contract in contracts:
			print(contract.text)

		#contracts[0].click()

	def get_contracts(self):
		contracts = []

		try:
			if not self.in_results_frame:
				wait=WebDriverWait(self.driver,300).until(EC.frame_to_be_available_and_switch_to_it((By.ID,'ifrFilterResults')))
				print("Found results iframe")
				self.in_results_frame=True
		except Exception as e:
			print(e)
			print("Couldn't get filter results frame")
			return []

		try:
			el=WebDriverWait(self.driver,300).until(
				EC.presence_of_all_elements_located((By.CLASS_NAME,"contract"))
			)
			print("Wait: located elements by css contract")
			count=len(el)
		except Exception as e:
			print(e)
			print("Couldn't get contracts elements")
			return []

		print("found "+str(len(el))+" contracts")

		for i in range(0,count):
			els=self.driver.find_elements_by_class_name("contract")
			
			contracts.append(els[i].text)

		return contracts

	def logout(self):
		return True
