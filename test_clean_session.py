import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

artifacts_dir = r"C:\Users\Admitrator\.gemini\antigravity-ide\brain\76f9668f-d18a-4f11-b36d-f92dee193e73"
img_path = os.path.join(artifacts_dir, ".user_uploaded", "media_1788286775435.png")

# 1. Create a fresh session
session_id = f"session_test_{int(time.time()*1000)}"
requests.post("http://127.0.0.1:8000/sessions", json={"id": session_id, "title": "Airport Vehicle Grounding Test"})

opts = Options()
opts.add_argument('--headless=new')
opts.add_argument('--window-size=1440,1050')
opts.add_argument('--disable-gpu')
opts.add_argument('--no-sandbox')
driver = webdriver.Chrome(options=opts)

try:
    driver.get(f"http://localhost:3010")
    time.sleep(2)
    
    # Click 'New Chat' or start clean
    new_chat_btn = driver.find_element(By.XPATH, "//button[contains(., 'New Analysis') or contains(@class, 'py-2.5')]")
    new_chat_btn.click()
    time.sleep(1)

    # Test 1: Deep Mode with Airport Image
    deep_btn = driver.find_element(By.XPATH, "//button[contains(@title, 'Deep Mode')]")
    deep_btn.click()
    time.sleep(1)

    file_input = driver.find_element(By.XPATH, "//input[@type='file']")
    file_input.send_keys(img_path)
    time.sleep(2)

    textarea = driver.find_element(By.TAG_NAME, "textarea")
    textarea.clear()
    textarea.send_keys("what vehicle can we see here in this image")
    textarea.send_keys(Keys.RETURN)
    print("Submitted Deep Mode query in clean session, waiting for response...")
    time.sleep(10)

    screenshot_deep_clean = os.path.join(artifacts_dir, "satquery_clean_deep_test.png")
    driver.save_screenshot(screenshot_deep_clean)
    print(f"Saved clean Deep Mode screenshot: {screenshot_deep_clean}")

    # Test 2: Quick Mode with Airport Image
    quick_btn = driver.find_element(By.XPATH, "//button[contains(@title, 'Quick Mode')]")
    quick_btn.click()
    time.sleep(1)

    file_input = driver.find_element(By.XPATH, "//input[@type='file']")
    file_input.send_keys(img_path)
    time.sleep(2)

    textarea = driver.find_element(By.TAG_NAME, "textarea")
    textarea.clear()
    textarea.send_keys("what vehicle can we see here in this image")
    textarea.send_keys(Keys.RETURN)
    print("Submitted Quick Mode query in clean session, waiting for response...")
    time.sleep(10)

    screenshot_quick_clean = os.path.join(artifacts_dir, "satquery_clean_quick_test.png")
    driver.save_screenshot(screenshot_quick_clean)
    print(f"Saved clean Quick Mode screenshot: {screenshot_quick_clean}")

finally:
    driver.quit()
    print("Clean session testing complete!")
