import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

def run_test():
    opts = Options()
    opts.add_argument('--headless=new')
    opts.add_argument('--window-size=1440,1100')
    driver = webdriver.Chrome(options=opts)
    artifacts_dir = r"C:\Users\Admitrator\.gemini\antigravity-ide\brain\76f9668f-d18a-4f11-b36d-f92dee193e73"

    try:
        driver.get("http://localhost:3010")
        time.sleep(3)

        # 1. Open Coords modal & apply region
        coords_btn = driver.find_element(By.XPATH, "//button[contains(., 'COORDS')]")
        coords_btn.click()
        time.sleep(2)

        preset_btn = driver.find_element(By.XPATH, "//button[contains(., 'ISRO Space Applications Centre') or contains(., 'Cartosat-2S')]")
        preset_btn.click()
        time.sleep(1)

        apply_btn = driver.find_element(By.XPATH, "//button[contains(., 'Apply Target Region')]")
        apply_btn.click()
        time.sleep(5)

        # 2. Type query character by character via send_keys so React onChange fires
        textarea = driver.find_element(By.TAG_NAME, "textarea")
        for char in "What has changed here in this sector over the past 5 years?":
            textarea.send_keys(char)
        time.sleep(1)

        # Submit via Enter
        textarea.send_keys(Keys.RETURN)
        print("Submitted prompt-driven bi-temporal change query. Waiting 30s...")
        time.sleep(30)

        screenshot_path = os.path.join(artifacts_dir, "prompt_driven_bitemporal_verified.png")
        driver.save_screenshot(screenshot_path)
        print("Saved verified screenshot:", screenshot_path)

    finally:
        driver.quit()

if __name__ == "__main__":
    run_test()
