import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def run_live_browser_tests():
    chrome_options = Options()
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--window-size=1440,1024')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    
    driver = webdriver.Chrome(options=chrome_options)
    artifacts_dir = r"C:\Users\Admitrator\.gemini\antigravity-ide\brain\76f9668f-d18a-4f11-b36d-f92dee193e73"
    img_path = os.path.join(artifacts_dir, ".user_uploaded", "media_1788286775435.png")

    try:
        print("Navigating to http://localhost:3010...")
        driver.get("http://localhost:3010")
        time.sleep(3)
        print("Page Title:", driver.title)
        
        # 1. Switch to Deep Mode
        deep_btn = driver.find_element(By.XPATH, "//button[contains(@title, 'Deep Mode')]")
        deep_btn.click()
        print("Selected Deep Thinking Mode.")
        time.sleep(1)

        # 2. Upload the airport image
        file_input = driver.find_element(By.XPATH, "//input[@type='file']")
        file_input.send_keys(img_path)
        print(f"Uploaded test image: {img_path}")
        time.sleep(2)

        # 3. Enter query in textarea
        textarea = driver.find_element(By.TAG_NAME, "textarea")
        textarea.clear()
        textarea.send_keys("what vehicle can we see here in this image")
        print("Entered query: 'what vehicle can we see here in this image'")
        time.sleep(1)

        # 4. Press Enter to submit
        textarea.send_keys(Keys.RETURN)
        print("Submitted query, waiting for AI response...")

        # Wait for response (up to 12s)
        time.sleep(10)
        
        # Save screenshot for Deep Mode
        screenshot_deep = os.path.join(artifacts_dir, "satquery_live_test_deep.png")
        driver.save_screenshot(screenshot_deep)
        print(f"Saved Deep Mode screenshot: {screenshot_deep}")

        # 5. Test Quick Mode
        quick_btn = driver.find_element(By.XPATH, "//button[contains(@title, 'Quick Mode')]")
        quick_btn.click()
        print("Switched to Quick Mode.")
        time.sleep(1)

        file_input = driver.find_element(By.XPATH, "//input[@type='file']")
        file_input.send_keys(img_path)
        time.sleep(2)

        textarea = driver.find_element(By.TAG_NAME, "textarea")
        textarea.clear()
        textarea.send_keys("what vehicle can we see here in this image")
        textarea.send_keys(Keys.RETURN)
        print("Submitted Quick Mode query, waiting for AI response...")

        time.sleep(10)

        screenshot_quick = os.path.join(artifacts_dir, "satquery_live_test_quick.png")
        driver.save_screenshot(screenshot_quick)
        print(f"Saved Quick Mode screenshot: {screenshot_quick}")

    except Exception as e:
        print(f"Error during browser test: {e}")
        driver.save_screenshot(os.path.join(artifacts_dir, "browser_error_screenshot.png"))
    finally:
        driver.quit()
        print("Browser testing complete!")

if __name__ == "__main__":
    run_live_browser_tests()
