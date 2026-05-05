import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
from typing import Dict, Any
import easyocr
import re

from app.services.layout_analyzer import LayoutResult

class OCREngine:
    def __init__(self):
        print("Loading OCR models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        if self.device == 'cpu':
            print("WARNING: No GPU detected - using CPU (slower)")
        
        try:
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            self.trocr_model.to(self.device)
            self.trocr_model.eval()
            print(f"TrOCR model loaded on {self.device}")
        except Exception as e:
            print(f"Warning: Could not load TrOCR: {e}")
            self.trocr_model = None
            self.trocr_processor = None
        
        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            print("EasyOCR model loaded")
        except Exception as e:
            print(f"Warning: Could not load EasyOCR: {e}")
            self.easyocr_reader = None
    
    def process(self, layout_result: LayoutResult) -> Dict[str, Any]:
        ocr_results = {
            'text_blocks': [],
            'diagrams': [],
            'structure': []
        }
        
        for idx in layout_result.reading_order:
            region = layout_result.text_regions[idx]
            x, y, w, h = region.bbox
            region_img = layout_result.image[y:y+h, x:x+w]
            text = self._ocr_region(region_img)
            
            if text:
                is_equation = self._is_equation(text)
                structure_info = self._detect_structure(text, region.region_type)
                
                ocr_results['text_blocks'].append({
                    'text': text,
                    'bbox': region.bbox,
                    'type': structure_info['type'],
                    'level': structure_info['level'],
                    'is_equation': is_equation,
                    'original_type': region.region_type
                })
        
        for diag in layout_result.diagram_regions:
            ocr_results['diagrams'].append({
                'bbox': diag.bbox,
                'image': diag.image
            })
        
        return ocr_results
    
    def _ocr_region(self, region_img: np.ndarray) -> str:
        if len(region_img.shape) == 3:
            pil_image = Image.fromarray(region_img).convert('RGB')
        else:
            pil_image = Image.fromarray(region_img).convert('L').convert('RGB')
        
        if self.trocr_model is not None and self.trocr_processor is not None:
            try:
                pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
                with torch.no_grad():
                    generated_ids = self.trocr_model.generate(pixel_values)
                text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                if text.strip():
                    return text.strip()
            except Exception as e:
                print(f"TrOCR error: {e}, falling back to EasyOCR")
        
        if self.easyocr_reader is not None:
            try:
                if isinstance(region_img, Image.Image):
                    region_img = np.array(region_img)
                results = self.easyocr_reader.readtext(region_img)
                text = " ".join([result[1] for result in results if result[2] > 0.5])
                return text.strip()
            except Exception as e:
                print(f"EasyOCR error: {e}")
        
        return ""
    
    def _is_equation(self, text: str) -> bool:
        equation_indicators = [
            r'[=+\-×÷*/]',
            r'[→←↑↓]',
            r'\d+[₀₁₂₃₄₅₆₇₈₉]',
            r'[A-Z][a-z]?\d+',
            r'\([a-z]+\)',
        ]
        
        equation_score = sum(1 for pattern in equation_indicators if re.search(pattern, text))
        
        if len(text.split()) <= 5 and equation_score >= 2:
            return True
        
        if '→' in text or '->' in text or '=' in text:
            if len([c for c in text if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']) >= 3:
                return True
        
        return False
    
    def _detect_structure(self, text: str, region_type: str) -> Dict[str, Any]:
        heading_indicators = [
            text.startswith('#'),
            text.isupper() and len(text) < 100,
            bool(re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]', text)),
            bool(re.match(r'^\d+\.?\s+[A-Z]', text)),
        ]
        
        if any(heading_indicators) or region_type == 'heading':
            if text.startswith('#'):
                level = len(re.match(r'#+', text).group())
            elif any(heading_indicators[:2]):
                level = 1
            else:
                level = 2
            return {'type': 'heading', 'level': min(level, 3)}
        
        if re.match(r'^[•·◦▪▫]\s', text) or re.match(r'^\(\w+\)\s', text) or re.match(r'^\d+[\.\)]\s', text):
            return {'type': 'list_item', 'level': 1}
        
        return {'type': 'paragraph', 'level': 0}
