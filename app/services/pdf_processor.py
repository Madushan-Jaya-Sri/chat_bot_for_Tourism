import pdfplumber
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    @staticmethod
    def extract_data_from_pdf(pdf_path: str) -> Tuple[str, List[List], List[str]]:
        try:
            text = ""
            tables = []
            image_paths = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text += page.extract_text() or ""
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
                    
                    for img_num, image in enumerate(page.images):
                        img = page.to_image()
                        img_path = Path(pdf_path).parent / f"page_{page_num}_img_{img_num}.png"
                        img.save(str(img_path))
                        image_paths.append(str(img_path))
            
            return text, tables, image_paths
        except Exception as e:
            logger.error(f"Error extracting data from PDF: {str(e)}")
            raise