"""
OCR fallback.

If PyMuPDF cannot extract text from a page,
this file converts the page into an image
and extracts text using RapidOCR.
"""

import fitz
from rapidocr_onnxruntime import RapidOCR
from PIL import Image
import io

# Load OCR model only once
ocr = RapidOCR()
def extract_page_with_ocr(pdf_path, page_number):
    """
    Extract text from one PDF page using OCR.
    Parameters
    ----------
    pdf_path : str
    page_number : int
    Returns
    -------
    str
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap(dpi=300)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    result, _ = ocr(image)
    doc.close()
    if result is None:
        return ""
    text = []
    for line in result:
        text.append(line[1])
    return "\n".join(text)