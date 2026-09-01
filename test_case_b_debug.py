import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

def run_debug_b():
    opts = Options()
    opts.add_argument('--headless=new')
    opts.add_argument('--window-size=1440,1100')
    driver = webdriver.Chrome(options=opts)
    artifacts_dir = r"C:\Users\Admitrator\.gemini\antigravity-ide\brain\76f9668f-d18a-4f11-b36d-f92dee193e73"

    try:
        driver.get("http://localhost:3010")
        time.sleep(3)

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
        textarea.send_keys("What has changed here in this sector over the past 5 years?")
        time.sleep(1)
        
        # Click the send button directly
        send_btn = driver.find_element(By.XPATH, "//button[contains(@class, 'bg-cyan-500') or @title='Send Query' or .//*[name()='svg' and contains(@class, 'lucide-send')]]")
        send_btn.click()
        print("Clicked Send button!")

        for i in range(8):
            time.sleep(5)
            print(f"Waited {(i+1)*5}s...")
            cards = driver.find_elements(By.XPATH, "//div[contains(., 'Satellite Imagery Observations')]")
            if cards:
                print("Found Satellite Imagery Observations card!")
                break

        time.sleep(2)
        screenshot_path = os.path.join(artifacts_dir, "debug_case_b_success.png")
        driver.save_screenshot(screenshot_path)
        print("Saved screenshot:", screenshot_path)

    finally:
        driver.quit()

if __name__ == "__main__":
    run_debug_b()
