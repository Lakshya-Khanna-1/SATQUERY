import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

def run_bitemporal_browser_test():
    opts = Options()
    opts.add_argument('--headless=new')
    opts.add_argument('--window-size=1440,1050')
    opts.add_argument('--disable-gpu')
    opts.add_argument('--no-sandbox')
    
    driver = webdriver.Chrome(options=opts)
    artifacts_dir = r"C:\Users\Admitrator\.gemini\antigravity-ide\brain\76f9668f-d18a-4f11-b36d-f92dee193e73"

    try:
        print("Navigating to http://localhost:3010...")
        driver.get("http://localhost:3010")
        time.sleep(3)

        # 1. Click 'New Analysis'
        new_btn = driver.find_element(By.XPATH, "//button[contains(., 'New Analysis') or contains(@class, 'py-2.5')]")
        new_btn.click()
        time.sleep(1)

        # 2. Click COORDS button
        coords_btn = driver.find_element(By.XPATH, "//button[contains(., 'COORDS')]")
        coords_btn.click()
        print("Opened Coords & Radius Modal.")
        time.sleep(2)

        # 3. Select a preset (e.g. ISRO SAC Ahmedabad or Cartosat-2S Urban)
        preset_btn = driver.find_element(By.XPATH, "//button[contains(., 'ISRO Space Applications Centre') or contains(., 'Cartosat-2S')]")
        preset_btn.click()
        print("Selected preset coordinate region.")
        time.sleep(1)

        # 4. Click 'Apply Target Region'
        apply_btn = driver.find_element(By.XPATH, "//button[contains(., 'Apply Target Region')]")
        apply_btn.click()
        print("Clicked Apply Target Region. Waiting for satellite rasters to stitch...")
        time.sleep(5)

        # 5. Send query
        textarea = driver.find_element(By.TAG_NAME, "textarea")
        textarea.send_keys(Keys.RETURN)
        print("Submitted Bi-Temporal Change Query, waiting for response...")
        time.sleep(10)

        # 6. Capture screenshot
        screenshot_path = os.path.join(artifacts_dir, "satquery_bitemporal_dual_screenshot.png")
        driver.save_screenshot(screenshot_path)
        print(f"Saved bi-temporal dual image screenshot: {screenshot_path}")

    except Exception as e:
        print(f"Error during bi-temporal browser test: {e}")
        driver.save_screenshot(os.path.join(artifacts_dir, "bitemporal_error_screenshot.png"))
    finally:
        driver.quit()
        print("Bi-temporal browser test complete!")

if __name__ == "__main__":
    run_bitemporal_browser_test()
