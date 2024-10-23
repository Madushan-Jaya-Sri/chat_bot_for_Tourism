from PIL import Image
import pytesseract
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    @staticmethod
    def analyze_images(image_paths: List[str]) -> List[Dict]:
        try:
            image_descriptions = []
            for img_path in image_paths:
                img = Image.open(img_path)
                text = pytesseract.image_to_string(img)
                image_descriptions.append({
                    "path": img_path,
                    "text": text.strip()
                })
            return image_descriptions
        except Exception as e:
            logger.error(f"Error analyzing images: {str(e)}")
            raise
