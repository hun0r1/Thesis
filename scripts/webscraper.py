import os
import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urljoin
from pathlib import Path

class RAW2KScraper:
    def __init__(self, base_path="D:\\Hunor\\EGYETEM\\A3.felev\\Allamvizsga+tesztek\\AutokReportok\\RAW2K"):
        self.base_url = "https://www.raw2k.co.uk"
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def get_auction_links(self, start_page=1, max_pages=None):
        """Get all auction links from the listing pages"""
        auction_links = []
        page = start_page
        
        while True:
            url = f"{self.base_url}/vehicle-auctions?drives=no&page={page}"
            print(f"\nFetching page {page}...")
            
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all auction cards/links
                # Looking for links that match the pattern /vehicle-auctions/YEAR-MAKE-MODEL-LOTNUM
                links = soup.find_all('a', href=re.compile(r'/vehicle-auctions/\d{4}-[a-z0-9-]+-\d+'))
                
                page_links = []
                for link in links:
                    href = link.get('href')
                    if href and href not in auction_links:
                        full_url = urljoin(self.base_url, href)
                        page_links.append(full_url)
                        auction_links.append(full_url)
                
                print(f"Found {len(page_links)} new auctions on page {page} (Total so far: {len(auction_links)})")
                
                # If no auctions found on this page, we've reached the end
                if not page_links:
                    print(f"No auctions found on page {page}. Stopping.")
                    break
                
                # Check if we've reached max_pages limit
                if max_pages and page >= max_pages:
                    print(f"Reached maximum page limit ({max_pages}). Stopping.")
                    break
                
                # Check for next page - look for pagination
                # Try multiple methods to detect if there's a next page
                has_next = False
                
                # Method 1: Look for "next" link
                next_button = soup.find('a', {'rel': 'next'})
                if next_button:
                    has_next = True
                
                # Method 2: Look for pagination with page numbers
                pagination = soup.find('ul', class_='pagination')
                if pagination:
                    page_links_in_pagination = pagination.find_all('a')
                    for p_link in page_links_in_pagination:
                        if p_link.get_text(strip=True).lower() in ['next', '>', '»']:
                            has_next = True
                            break
                        # Check if there's a link to the next page number
                        try:
                            page_num = int(p_link.get_text(strip=True))
                            if page_num == page + 1:
                                has_next = True
                                break
                        except:
                            pass
                
                # Method 3: Just try the next page if we found auctions on this one
                # For sites without clear pagination, keep going until we hit an empty page
                if page_links:
                    has_next = True
                
                if not has_next:
                    print(f"No next page found. Stopping at page {page}.")
                    break
                
                page += 1
                time.sleep(1)  # Be polite to the server
                
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                break
        
        return list(set(auction_links))  # Remove duplicates
    
    def extract_vehicle_info(self, soup):
        """Extract vehicle information from the page"""
        # Get vehicle title
        title_elem = soup.find('h1', itemprop='name')
        title = title_elem.text.strip() if title_elem else "Unknown Vehicle"
        
        # Get lot number
        lot_elem = soup.find('span', class_='lime-text bold', itemprop='productID')
        lot_number = lot_elem.text.strip() if lot_elem else "Unknown"
        
        # Get damage report - using itemprop attribute
        damage_report = "Not available"
        damage_elem = soup.find('p', {'itemprop': 'knownVehicleDamages'})
        if damage_elem:
            damage_report = damage_elem.get_text(strip=True)
        else:
            # Alternative: look for "Damage report:" header
            all_text = soup.get_text()
            if 'Damage report:' in all_text:
                # Try to extract it from the raw text
                damage_idx = all_text.find('Damage report:')
                if damage_idx != -1:
                    # Get text after "Damage report:" until next section
                    text_after = all_text[damage_idx + len('Damage report:'):damage_idx + 1000]
                    # Try to find where it ends (usually at next header or double newline)
                    end_markers = ['Extra information:', 'Viewing:', 'Delivery:', '\n\n\n']
                    end_idx = len(text_after)
                    for marker in end_markers:
                        idx = text_after.find(marker)
                        if idx != -1 and idx < end_idx:
                            end_idx = idx
                    damage_report = text_after[:end_idx].strip()
        
        # Get extra information
        extra_info = "Not available"
        extra_info_headers = soup.find_all('h5')
        for header in extra_info_headers:
            if 'Extra information' in header.get_text():
                extra_div = header.find_next('p')
                if extra_div:
                    extra_info = extra_div.get_text(strip=True)
                break
        
        # Get viewing info
        viewing_info = "Not available"
        viewing_headers = soup.find_all('h5')
        for header in viewing_headers:
            if 'Viewing' in header.get_text():
                viewing_p = header.find_next('p')
                if viewing_p:
                    viewing_info = viewing_p.get_text(strip=True)
                break
        
        # Get delivery info
        delivery_info = "Not available"
        delivery_headers = soup.find_all('h5')
        for header in delivery_headers:
            if 'Delivery' in header.get_text():
                delivery_p = header.find_next('p')
                if delivery_p:
                    delivery_info = delivery_p.get_text(strip=True)
                break
        
        # Get all key facts from the grid
        key_facts = {}
        grid_cells = soup.find_all('div', class_='cell')
        for cell in grid_cells:
            text = cell.get_text(strip=True)
            key_facts[len(key_facts)] = text
        
        # Clean title for folder name (remove invalid characters)
        clean_title = re.sub(r'[<>:"/\\|?*]', '_', title)
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()
        
        return {
            'title': title,
            'lot_number': lot_number,
            'folder_name': f"{clean_title}_LOT{lot_number}",
            'damage_report': damage_report,
            'extra_info': extra_info,
            'viewing_info': viewing_info,
            'delivery_info': delivery_info,
            'key_facts': key_facts
        }
    
    def get_all_image_urls(self, soup):
        """Extract all image URLs from the auction page"""
        image_urls = []
        
        # Find all lightbox links (data-lightbox="car")
        lightbox_links = soup.find_all('a', {'data-lightbox': 'car'})
        
        for link in lightbox_links:
            href = link.get('href')
            if href and href.startswith('http'):
                image_urls.append(href)
        
        # Also check for images in img tags with specific patterns
        img_tags = soup.find_all('img', class_='vehicle-image')
        for img in img_tags:
            src = img.get('src')
            if src and src.startswith('http') and src not in image_urls:
                image_urls.append(src)
        
        return image_urls
    
    def download_image(self, url, save_path):
        """Download a single image"""
        try:
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        except Exception as e:
            print(f"  Error downloading image: {e}")
            return False
    
    def scrape_auction(self, auction_url):
        """Scrape a single auction page and download all images"""
        print(f"\nProcessing: {auction_url}")
        
        try:
            response = self.session.get(auction_url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract vehicle info
            vehicle_info = self.extract_vehicle_info(soup)
            print(f"  Vehicle: {vehicle_info['title']}")
            print(f"  Lot Number: {vehicle_info['lot_number']}")
            
            # Create folder for this vehicle
            vehicle_folder = self.base_path / vehicle_info['folder_name']
            vehicle_folder.mkdir(parents=True, exist_ok=True)
            
            # Get all image URLs
            image_urls = self.get_all_image_urls(soup)
            print(f"  Found {len(image_urls)} images")
            
            # Download all images
            for idx, img_url in enumerate(image_urls, 1):
                # Extract filename from URL
                img_filename = os.path.basename(img_url.split('?')[0])
                if not img_filename:
                    img_filename = f"image_{idx:02d}.jpg"
                
                save_path = vehicle_folder / img_filename
                
                if save_path.exists():
                    print(f"  [{idx}/{len(image_urls)}] Already exists: {img_filename}")
                else:
                    print(f"  [{idx}/{len(image_urls)}] Downloading: {img_filename}")
                    if self.download_image(img_url, save_path):
                        print(f"  [{idx}/{len(image_urls)}] Saved: {img_filename}")
                
                time.sleep(0.5)  # Be polite to the server
            
            # Save auction info to text file
            info_file = vehicle_folder / "auction_info.txt"
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: {vehicle_info['title']}\n")
                f.write(f"Lot Number: {vehicle_info['lot_number']}\n")
                f.write(f"URL: {auction_url}\n")
                f.write(f"Images Downloaded: {len(image_urls)}\n\n")
                
                # Key facts
                if vehicle_info['key_facts']:
                    f.write("=" * 60 + "\n")
                    f.write("KEY FACTS\n")
                    f.write("=" * 60 + "\n")
                    for fact in vehicle_info['key_facts'].values():
                        f.write(f"• {fact}\n")
                    f.write("\n")
                
                f.write("=" * 60 + "\n")
                f.write("DAMAGE REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"{vehicle_info['damage_report']}\n\n")
                
                if vehicle_info['extra_info'] != "Not available":
                    f.write("=" * 60 + "\n")
                    f.write("EXTRA INFORMATION\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"{vehicle_info['extra_info']}\n\n")
                
                if vehicle_info['viewing_info'] != "Not available":
                    f.write("=" * 60 + "\n")
                    f.write("VIEWING INFORMATION\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"{vehicle_info['viewing_info']}\n\n")
                
                if vehicle_info['delivery_info'] != "Not available":
                    f.write("=" * 60 + "\n")
                    f.write("DELIVERY INFORMATION\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"{vehicle_info['delivery_info']}\n")
            
            # Save damage report separately ONLY if it exists
            if vehicle_info['damage_report'] != "Not available":
                damage_file = vehicle_folder / "damage_report.txt"
                with open(damage_file, 'w', encoding='utf-8') as f:
                    f.write(vehicle_info['damage_report'])
                print(f"  ✓ Damage report saved")
            else:
                print(f"  ⚠ No damage report found")
            
            return True
            
        except Exception as e:
            print(f"  Error processing auction: {e}")
            return False
    
    def run(self, start_page=1, max_pages=None):
        """Main scraping function"""
        print("=" * 60)
        print("RAW2K Auction Scraper")
        print("=" * 60)
        print(f"Save path: {self.base_path}")
        print(f"Starting from page: {start_page}")
        if max_pages:
            print(f"Maximum pages: {max_pages}")
        print("=" * 60)
        
        # Get all auction links
        print("\nStep 1: Collecting auction links...")
        auction_links = self.get_auction_links(start_page, max_pages)
        print(f"\nFound {len(auction_links)} total auctions")
        
        # Process each auction
        print("\nStep 2: Processing auctions and downloading images...")
        successful = 0
        failed = 0
        
        for idx, auction_url in enumerate(auction_links, 1):
            print(f"\n[{idx}/{len(auction_links)}]")
            if self.scrape_auction(auction_url):
                successful += 1
            else:
                failed += 1
            
            time.sleep(1)  # Be polite between auctions
        
        # Summary
        print("\n" + "=" * 60)
        print("SCRAPING COMPLETE")
        print("=" * 60)
        print(f"Total auctions: {len(auction_links)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Save location: {self.base_path}")
        print("=" * 60)


if __name__ == "__main__":
    # Create scraper instance
    scraper = RAW2KScraper()
    
    # Run the scraper
    # You can customize:
    # - start_page: which page to start from (default: 1)
    # - max_pages: maximum number of pages to scrape (default: None = all pages)
    
    # Example: Scrape all pages
    scraper.run()
    
    # Example: Scrape only first 3 pages
    # scraper.run(start_page=1, max_pages=3)
    
    # Example: Resume from page 5
    # scraper.run(start_page=5)