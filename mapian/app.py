import mss
import cv2
import numpy as np
import pytesseract
import threading
import time
from datetime import datetime

# Set Tesseract path (adjust if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Default Windows path

class MiniAladdinScreen:
    def __init__(self):
        self.sct = mss.mss()
        # Define Brave window region (adjust these coordinates to match your Brave window)
        self.region = {"top": 50, "left": 50, "width": 1000, "height": 700}  # Adjusted for your screenshot
        self.current_price = 0.0
        self.recent_prices = []
        self.recent_volumes = []  # Estimated from chart bars
        self.running = True
        self.analysis_thread = threading.Thread(target=self.background_analysis, daemon=True)
        self.analysis_thread.start()

    def capture_screen(self):
        """Capture the Brave browser window."""
        try:
            screenshot = self.sct.grab(self.region)
            img = np.array(screenshot)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f"Capture Error: {e}")
            return None

    def extract_price(self, img):
        """Extract bid price from the chart using OCR with debug output."""
        if img is None:
            return self.current_price
        # Define ROI where bid price appears (top-right corner, adjusted for your layout)
        price_roi = img[50:100, 800:950]  # Top-right, capturing "2,900.117"
        gray = cv2.cvtColor(price_roi, cv2.COLOR_BGR2GRAY)
        # Enhance contrast for OCR
        gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Add blur for better OCR
        text = pytesseract.image_to_string(gray, config='--psm 7 --oem 3')  # Single line, neural net
        print(f"OCR Output: '{text}'")  # Debug: show what OCR reads
        try:
            # Clean text to get numeric price (handle commas)
            cleaned_text = ''.join(filter(lambda x: x.isdigit() or x == '.', text.replace(',', '')))
            price = float(cleaned_text) if cleaned_text else self.current_price
            return price
        except ValueError:
            print(f"Price Parsing Failed: Using last price ${self.current_price:.2f}")
            return self.current_price  # Fallback to last price if OCR fails

    def extract_volume(self, img):
        """Estimate volume from chart bars (simplified) with debug output."""
        if img is None:
            return 0.0
        # Define ROI where volume bars appear (bottom of chart)
        volume_roi = img[550:650, 50:950]  # Bottom area with volume bars
        gray = cv2.cvtColor(volume_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bar_height = np.sum(thresh) / 255  # Rough pixel count as volume proxy
        print(f"Volume Pixels: {bar_height}")  # Debug: show raw pixel count
        return bar_height / 1000  # Scale to a reasonable range

    def predict_next_move(self):
        """Predict next intraday move based on price trend."""
        if len(self.recent_prices) < 3:
            return "Hold - Insufficient data"
        trend = np.mean(np.diff(self.recent_prices[-3:]))
        if trend > 0.1:  # Price rising
            return "Buy - Uptrend detected"
        elif trend < -0.1:  # Price falling
            return "Sell - Downtrend detected"
        return "Hold - No clear trend"

    def background_analysis(self):
        """Analyze for big shark entries in the background."""
        while self.running:
            if len(self.recent_prices) >= 5 and len(self.recent_volumes) >= 5:
                price_change = abs(self.recent_prices[-1] - self.recent_prices[-2])
                vol_change = abs(self.recent_volumes[-1] - np.mean(self.recent_volumes[:-1]))
                if price_change > 0.5 or vol_change > 1.0:  # Adjustable thresholds
                    print(f"🚨 Big Shark Alert: Price jump: ${price_change:.2f}, Volume spike: {vol_change:.2f}")
            time.sleep(1)

    def run(self):
        """Main loop to capture and analyze the chart."""
        print("Starting Mini-Aladdin Screen Trader...")
        print("Open your XAU/USD chart in Brave and adjust region if needed.")
        while self.running:
            img = self.capture_screen()
            if img is None:
                print("Skipping frame due to capture error.")
                time.sleep(0.5)  # Wait before retrying
                continue
            self.current_price = self.extract_price(img)
            volume = self.extract_volume(img)
            
            self.recent_prices.append(self.current_price)
            self.recent_volumes.append(volume)
            if len(self.recent_prices) > 5:
                self.recent_prices.pop(0)
            if len(self.recent_volumes) > 5:
                self.recent_volumes.pop(0)

            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"XAU/USD Price: ${self.current_price:.2f}")
            print(f"Estimated Volume: {volume:.2f} (proxy)")
            print(f"Next Move: {self.predict_next_move()}")
            print("-" * 50)
            
            # Show captured image (for debugging)
            cv2.imshow("Chart", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
            
            time.sleep(0.5)  # Slow down to 500ms updates for stability

        self.running = False
        cv2.destroyAllWindows()

if __name__ == "__main__":
    trader = MiniAladdinScreen()
    try:
        trader.run()
    except KeyboardInterrupt:
        trader.running = False
        print("Trading stopped.")