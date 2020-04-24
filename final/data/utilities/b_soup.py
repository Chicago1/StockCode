from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

import bs4
import time

driver = webdriver.Chrome(ChromeDriverManager().install()) #install the webdriver

driver.get('https://investor.vanguard.com/etf/profile/VOX') #link to use

time.sleep(5) #NEEDED for the javascript to run in the browser and make the tables

elem = driver.find_element_by_xpath("//*") #extracts the whole HTML file
source_code = elem.get_attribute("outerHTML")


cleanPageData = bs4.BeautifulSoup(source_code, 'html.parser') #parses the HTML

cleanerPageData = cleanPageData.find_all('td', class_="ng-scope ng-binding fixedCol")[8:] #skips the first 8 data vectors

wordList = []
for hit in cleanerPageData:
    hit = hit.text.strip()
    split_string = hit.split("&", 1)
    substring = split_string[0]

    if substring not in wordList:
        wordList.append(substring)

print(wordList)
    
driver.close()