from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

opts = Options()
opts.add_argument('--headless=new')
opts.add_argument('--window-size=1440,1024')
driver = webdriver.Chrome(options=opts)
driver.get('http://localhost:3010')
time.sleep(3)

print('--- BUTTONS FOUND ---')
buttons = driver.find_elements(By.TAG_NAME, 'button')
for b in buttons:
    text = b.text.strip().replace('\n', ' ')
    cls = (b.get_attribute('class') or '')[:40]
    print(f"Button: text='{text}', class='{cls}'")

print('\n--- INPUTS FOUND ---')
inputs = driver.find_elements(By.TAG_NAME, 'input')
for inp in inputs:
    t = inp.get_attribute('type') or ''
    p = inp.get_attribute('placeholder') or ''
    print(f"Input: type='{t}', placeholder='{p}'")

driver.quit()
