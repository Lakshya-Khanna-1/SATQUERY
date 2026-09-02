import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

def run_sidebar_test():
    opts = Options()
    opts.add_argument('--headless=new')
    opts.add_argument('--window-size=1440,1050')
    driver = webdriver.Chrome(options=opts)
    artifacts_dir = r"C:\Users\Admitrator\.gemini\antigravity-ide\brain\76f9668f-d18a-4f11-b36d-f92dee193e73"

    try:
        driver.get("http://localhost:3010")
        time.sleep(3)

        # 1. Capture Collapsed Sidebar Screenshot (default state when not opened)
        screenshot_collapsed = os.path.join(artifacts_dir, "sidebar_collapsed_test.png")
        driver.save_screenshot(screenshot_collapsed)
        print("Saved Collapsed Sidebar screenshot:", screenshot_collapsed)

        # 2. Expand Sidebar
        expand_btn = driver.find_element(By.XPATH, "//button[contains(@title, 'Expand Sidebar') or contains(@title, 'Collapse Sidebar')]")
        expand_btn.click()
        time.sleep(1)

        # 3. Capture Expanded Sidebar Screenshot
        screenshot_expanded = os.path.join(artifacts_dir, "sidebar_expanded_test.png")
        driver.save_screenshot(screenshot_expanded)
        print("Saved Expanded Sidebar screenshot:", screenshot_expanded)

    finally:
        driver.quit()

if __name__ == "__main__":
    run_sidebar_test()
