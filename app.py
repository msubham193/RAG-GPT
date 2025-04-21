import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin

URLS = [
    "https://www.cime.ac.in/about-cime.html",
    "https://www.cime.ac.in/vision-mission.html",
    "https://www.cime.ac.in/principal-s-message-page.html",
    "https://www.cime.ac.in/administrative-staff.html",
    "https://www.cime.ac.in/admission-procedure.html",
    "https://www.cime.ac.in/admission-notice.html",
    "https://www.cime.ac.in/mba-fee-structure.html",
    "https://www.cime.ac.in/fee-structure-mca.html",
    "https://www.cime.ac.in/fee-structure-msc-cs-.html",
    "https://www.cime.ac.in/seat-details.html",
    "https://www.cime.ac.in/faculties-mba.html",
    "https://www.cime.ac.in/faculties-mca.html",
    "https://www.cime.ac.in/digital-classrooms.html",
    "https://www.cime.ac.in/board-room.html",
    "https://www.cime.ac.in/huge-computer-lab.html",
    "https://www.cime.ac.in/library.html",
    "https://www.cime.ac.in/playground.html",
    "https://www.cime.ac.in/hostel.html",
    "https://www.cime.ac.in/placement-section.html",
    "https://www.cime.ac.in/canteen.html",
    "https://www.cime.ac.in/library-reading-area.html",
    "https://www.cime.ac.in/anti-ragging-helpline.html",
    "https://www.cime.ac.in/"
]

output_file = "data.txt"

def scrape_page(url):
    """Scrapes content and image paths from a given page and saves them."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        text = soup.get_text(separator="\n", strip=True)
        
        image_urls = []
        for img in soup.find_all("img", src=True):
            img_url = urljoin(url, img["src"])
            image_urls.append(img_url)
        
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\nURL: {url}\n{'='*80}\n")
            f.write(text + "\n")
            if image_urls:
                f.write("\nImages:\n" + "\n".join(image_urls) + "\n")
    except requests.RequestException as e:
        print(f"Failed to scrape {url}: {e}")

if __name__ == "__main__":
    open(output_file, "w").close()  # Clear the file before scraping
    for url in URLS:
        print(f"Scraping: {url}")
        scrape_page(url)
        time.sleep(1)  # Be respectful, avoid hammering the server
    print("Scraping completed! Data saved in data.txt.")
