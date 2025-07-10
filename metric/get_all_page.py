from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import os
import time
import re

# TODO: Capture the page screenshot and save it to the local machine.
def capture_all_pages(url, save_dir):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def get_title_text():
        try:
            # Get the textContent of the first div under the element with id 'chat' (i.e., titleDiv)
            title = driver.execute_script('return document.querySelector("#chat > div").textContent')
            # Remove special characters, keep only Chinese, English, numbers, and underscores
            title = re.sub(r'[^\w\u4e00-\u9fa5-]', '_', title)
            return title.strip('_') or 'untitled'
        except Exception:
            return f'untitled'

    page_num = 1
    while True:
        # Get page width and height
        width = driver.execute_script("return document.body.scrollWidth")
        height = driver.execute_script("return document.body.scrollHeight")
        driver.set_window_size(540, height+200)
        time.sleep(0.5)  # Wait for rendering
        title = get_title_text()
        save_path = os.path.join(save_dir, f"{title}_random.png")
        driver.save_screenshot(save_path)
        # Check if the "Next" button is enabled
        try:
            next_btn = driver.find_element('xpath', "//button[text()='Next']")
            if next_btn.is_enabled():
                next_btn.click()
                time.sleep(0.5)  # Wait for page switch
                page_num += 1
            else:
                break
        except Exception:
            break
    driver.quit()

if __name__ == "__main__":
    capture_all_pages("http://localhost:8080/view-contact/", "role_based_12_turns_dialogs_with_images_full_random")
