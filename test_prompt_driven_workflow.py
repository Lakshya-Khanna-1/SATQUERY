import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def run_prompt_driven_tests():
    opts = Options()
    opts.add_argument('--headless=new')
    opts.add_argument('--window-size=1440,1100')
    driver = webdriver.Chrome(options=opts)
    artifacts_dir = r"C:\Users\Admitrator\.gemini\antigravity-ide\brain\76f9668f-d18a-4f11-b36d-f92dee193e73"

    try:
        # TEST CASE A: Map selection with Single Image question for Specialist A
        print("--- Running Test Case A: Coords map image -> Specialist A prompt ---")
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

        # Click Apply
        apply_btn = driver.find_element(By.XPATH, "//button[contains(., 'Apply Target Region')]")
        apply_btn.click()
        time.sleep(5)

        # Type Specialist A single-image query
        textarea = driver.find_element(By.TAG_NAME, "textarea")
        textarea.send_keys("What primary geographic and urban features can be seen in this satellite image?")
        textarea.send_keys(Keys.RETURN)
        print("Submitted Specialist A query, waiting for response...")

        wait = WebDriverWait(driver, 30)
        wait.until(EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Single Optical Scene') or contains(text(), 'Satellite Imagery')]")))
        time.sleep(3)

        screenshot_a = os.path.join(artifacts_dir, "coords_specialist_a_test.png")
        driver.save_screenshot(screenshot_a)
        print("Saved Test Case A screenshot:", screenshot_a)

        # TEST CASE B: Map selection with Bi-Temporal Change question for Specialist B
        print("\n--- Running Test Case B: Coords map image -> Specialist B prompt ---")
        new_btn = driver.find_element(By.XPATH, "//button[contains(., 'New Analysis') or contains(@class, 'py-2.5')]")
        new_btn.click()
        time.sleep(2)

        coords_btn = driver.find_element(By.XPATH, "//button[contains(., 'COORDS')]")
        coords_btn.click()
        time.sleep(2)

        preset_btn = driver.find_element(By.XPATH, "//button[contains(., 'ISRO Space Applications Centre') or contains(., 'Cartosat-2S')]")
        preset_btn.click()
        time.sleep(1)

        apply_btn = driver.find_element(By.XPATH, "//button[contains(., 'Apply Target Region')]")
        apply_btn.click()
        time.sleep(5)

        # Type Specialist B bi-temporal query
        textarea = driver.find_element(By.TAG_NAME, "textarea")
        textarea.send_keys("What has changed here in this sector over the past 5 years?")
        textarea.send_keys(Keys.RETURN)
        print("Submitted Specialist B change query, waiting for bi-temporal auto-fetch & response...")

        wait.until(EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Bi-Temporal Pair')]")))
        time.sleep(3)

        screenshot_b = os.path.join(artifacts_dir, "coords_specialist_b_prompt_driven_test.png")
        driver.save_screenshot(screenshot_b)
        print("Saved Test Case B screenshot:", screenshot_b)

    finally:
        driver.quit()
        print("\nAll prompt-driven tests complete!")

if __name__ == "__main__":
    run_prompt_driven_tests()
