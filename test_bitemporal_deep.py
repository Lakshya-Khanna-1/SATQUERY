import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def run_deep_bitemporal_test():
    opts = Options()
    opts.add_argument('--headless=new')
    opts.add_argument('--window-size=1440,1100')
    driver = webdriver.Chrome(options=opts)
    artifacts_dir = r"C:\Users\Admitrator\.gemini\antigravity-ide\brain\76f9668f-d18a-4f11-b36d-f92dee193e73"

    try:
        driver.get("http://localhost:3010")
        time.sleep(3)

        # 1. Switch to Deep Mode
        deep_btn = driver.find_element(By.XPATH, "//button[contains(@title, 'Deep Mode')]")
        deep_btn.click()
        time.sleep(1)

        # 2. Open Coords modal
        coords_btn = driver.find_element(By.XPATH, "//button[contains(., 'COORDS')]")
        coords_btn.click()
        time.sleep(2)

        preset_btn = driver.find_element(By.XPATH, "//button[contains(., 'ISRO Space Applications Centre') or contains(., 'Cartosat-2S')]")
        preset_btn.click()
        time.sleep(1)

        apply_btn = driver.find_element(By.XPATH, "//button[contains(., 'Apply Target Region')]")
        apply_btn.click()
        time.sleep(5)

        textarea = driver.find_element(By.TAG_NAME, "textarea")
        textarea.send_keys(Keys.RETURN)
        print("Submitted query, waiting for response...")

        # Wait until the bot response arrives (up to 30s)
        wait = WebDriverWait(driver, 30)
        wait.until(EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Satellite Imagery Observations')]")))
        time.sleep(2)

        screenshot_path = os.path.join(artifacts_dir, "satquery_bitemporal_deep_dual_screenshot.png")
        driver.save_screenshot(screenshot_path)
        print("Saved deep bi-temporal screenshot:", screenshot_path)
    finally:
        driver.quit()

if __name__ == "__main__":
    run_deep_bitemporal_test()
