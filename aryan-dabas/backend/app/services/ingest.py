import os
from PIL import Image
import pytesseract

# Configure Tesseract path - update this path as needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Try different ways to import PyMuPDF
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import pymupdf as fitz  # Alternative import
        PYMUPDF_AVAILABLE = True
    except ImportError:
        print("Warning: PyMuPDF not available. PDF processing will be disabled.")
        PYMUPDF_AVAILABLE = False

def extract_text_from_image(image_path):
    """Extract text from image using OCR"""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error extracting text from image {image_path}: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    if not PYMUPDF_AVAILABLE:
        print("PyMuPDF not available. Cannot process PDF files.")
        return ""
    
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

def extract_text_from_pdf_alternative(pdf_path):
    """Alternative PDF text extraction using PyPDF2 (fallback)"""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except ImportError:
        print("PyPDF2 not available as fallback")
        return ""
    except Exception as e:
        print(f"Error with PyPDF2 extraction: {e}")
        return ""

def process_files(input_dir, output_dir):
    """Process all files in input directory and save extracted text to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist")
        return
    
    processed_files = []
    
    for filename in os.listdir(input_dir):
        full_path = os.path.join(input_dir, filename)
        text = ""
        
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            print(f"Processing image: {filename}")
            text = extract_text_from_image(full_path)
            
        elif filename.lower().endswith(".pdf"):
            print(f"Processing PDF: {filename}")
            text = extract_text_from_pdf(full_path)
            
            # Try alternative method if first one fails
            if not text.strip():
                print(f"Trying alternative PDF extraction for {filename}")
                text = extract_text_from_pdf_alternative(full_path)
        else:
            continue
        
        if text.strip():
            out_file = os.path.join(output_dir, filename + ".txt")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(text.strip())
            processed_files.append(filename)
            print(f"Successfully processed: {filename}")
        else:
            print(f"No text extracted from: {filename}")
    
    print(f"Processing complete. {len(processed_files)} files processed.")
    return processed_files

def test_imports():
    """Test function to check if all required modules are available"""
    print("Testing imports...")
    
    # Test Tesseract
    try:
        test_image = Image.new('RGB', (100, 30), color='white')
        pytesseract.image_to_string(test_image)
        print("✓ Tesseract OCR working")
    except Exception as e:
        print(f"✗ Tesseract OCR error: {e}")
    
    # Test PyMuPDF
    if PYMUPDF_AVAILABLE:
        try:
            # Test with a simple operation
            print(f"✓ PyMuPDF available (version: {fitz.version})")
        except Exception as e:
            print(f"✗ PyMuPDF error: {e}")
    else:
        print("✗ PyMuPDF not available")

if __name__ == "__main__":
    test_imports()