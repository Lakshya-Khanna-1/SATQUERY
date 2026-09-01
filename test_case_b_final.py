import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_case_b_final():
    opts = Options()
    opts.add_argument('--headless=new')
    opts.add_argument('--window-size=1440,1100')
    driver = webdriver.Chrome(options=opts)
    artifacts_dir = r"C:\Users\Admitrator\.gemini\antigravity-ide\brain\76f9668f-d18a-4f11-b36d-f92dee193e73"

    try:
        driver.get("http://localhost:3010")
        time.sleep(3)

        # Open Coords modal
        coords_btn = driver.find_element(By.XPATH, "//button[contains(., 'COORDS')]")
        coords_btn.click()
        time.sleep(2)

        # Pick preset
        preset_btn = driver.find_element(By.XPATH, "//button[contains(., 'ISRO Space Applications Centre') or contains(., 'Cartosat-2S')]")
        preset_btn.click()
        time.sleep(1)

        # Click Apply Target Region (attaches 1 present image)
        apply_btn = driver.find_element(By.XPATH, "//button[contains(., 'Apply Target Region')]")
        apply_btn.click()
        time.sleep(5)

        # Send bi-temporal change prompt
        textarea = driver.find_element(By.TAG_NAME, "textarea")
        textarea.send_keys("What has changed here in this sector over the past 5 years?")
        textarea.send_keys(Keys.RETURN)
        print("Submitted change prompt, waiting for auto-fetch & response...")

        # Wait until bot finishes responding (message card appears)
        wait = WebDriverWait(driver, 45)
        wait.until(EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Bi-Temporal Pair')]")))
        time.sleep(2)

        screenshot_b = os.path.join(artifacts_dir, "coords_specialist_b_prompt_driven_test.png")
        driver.save_screenshot(screenshot_b)
        print("Saved finished Test Case B screenshot:", screenshot_b)
    finally:
        driver.quit()

if __name__ == "__main__":
    test_case_b_final()
