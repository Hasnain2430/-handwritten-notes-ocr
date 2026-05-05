"""
Qwen-VL-OCR service for handwritten notes OCR.
Uses Alibaba Cloud Model Studio API with OpenAI-compatible endpoint.

ROLE: PRIMARY AND EXCLUSIVE OCR ENGINE
- Extracts editable text from handwritten notes
- Detects and extracts mathematical equations as LaTeX
- Identifies diagram regions (excluded from text OCR)
- Preserves document structure (headings, paragraphs, lists)

NO FALLBACK: If Qwen-VL-OCR is unavailable, processing HALTS with error.

API CONFIGURATION:
- Model: qwen-vl-ocr (exact name - NO substitution to other Qwen-VL models)
- Endpoint: Alibaba Cloud Model Studio (international)
"""
import os
import logging
import base64
import re
import html
from typing import Optional, Tuple, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# ALIBABA CLOUD MODEL STUDIO API CONFIGURATION
# ============================================================
# International endpoint (Singapore region)
ALIBABA_API_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# CRITICAL: Use exact model name - NO substitution to other Qwen-VL or Qwen3 models
QWEN_VL_OCR_MODEL = "qwen-vl-ocr"


class QwenVLOCR:
    """
    Qwen-VL-OCR service for handwritten notes recognition.
    
    Uses Alibaba Cloud Model Studio API with OpenAI-compatible interface.
    
    CRITICAL: This is the PRIMARY and EXCLUSIVE OCR engine.
    NO fallback to Gemini or other OCR engines for text extraction.
    """
    
    def __init__(self):
        """Initialize Qwen-VL-OCR service."""
        self.api_key = os.getenv('ALIBABA_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
        self.available = False
        self.client = None
        self.initialization_error = None
        
        if not self.api_key:
            self.initialization_error = (
                "ALIBABA_API_KEY or DASHSCOPE_API_KEY not set. "
                "Set either environment variable to enable Qwen-VL-OCR."
            )
            logger.error(f"❌ {self.initialization_error}")
            return
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=ALIBABA_API_BASE_URL
            )
            self.available = True
            logger.info(f"✅ Qwen-VL-OCR initialized (model: {QWEN_VL_OCR_MODEL})")
            logger.info(f"   API endpoint: {ALIBABA_API_BASE_URL}")
            
        except ImportError:
            self.initialization_error = "OpenAI package not installed. Install with: pip install openai"
            logger.error(f"❌ {self.initialization_error}")
            self.available = False
        except Exception as e:
            self.initialization_error = f"Qwen-VL-OCR initialization failed: {e}"
            logger.error(f"❌ {self.initialization_error}")
            self.available = False

    
    def extract_text_from_image(
        self,
        image_path: str,
        image_width: int = None,
        image_height: int = None
    ) -> Tuple[str, List[Dict]]:
        """
        Extract text, equations, and detect diagrams from an image.
        
        CRITICAL: This is the ONLY OCR method. No fallback.
        
        Args:
            image_path: Path to the image file
            image_width: Width of the image (for diagram position estimation)
            image_height: Height of the image (for diagram position estimation)
            
        Returns:
            Tuple of (structured_text, diagram_regions)
            - structured_text: Extracted text with headings and LaTeX equations
            - diagram_regions: List of detected diagram regions for cropping
            
        Raises:
            RuntimeError: If Qwen-VL-OCR is unavailable or API fails
        """
        if not self.available:
            error_msg = self.initialization_error or (
                "CRITICAL: Qwen-VL-OCR is unavailable. "
                "Set ALIBABA_API_KEY or DASHSCOPE_API_KEY environment variable. "
                "OCR processing HALTED - no fallback available."
            )
            raise RuntimeError(error_msg)
        
        try:
            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Determine MIME type
            ext = Path(image_path).suffix.lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_types.get(ext, 'image/jpeg')
            
            # Build OCR prompt
            prompt = self._get_ocr_prompt()
            
            logger.info(f"   📤 Sending image to Qwen-VL-OCR (model: {QWEN_VL_OCR_MODEL})...")
            
            # Call Qwen-VL-OCR API
            response = self.client.chat.completions.create(
                model=QWEN_VL_OCR_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                },
                                "min_pixels": 3136,  # Minimum input resolution (required by API)
                                "max_pixels": 32 * 32 * 8192  # Maximum input resolution
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=4096
            )
            
            # VERIFICATION: Check that response is valid and from the correct model
            if not response:
                raise RuntimeError("Qwen-VL-OCR API returned empty response. OCR processing HALTED.")
            
            # Log model used (if available in response)
            if hasattr(response, 'model'):
                actual_model = response.model
                logger.info(f"   🔍 Response from model: {actual_model}")
                # Verify it's the expected model
                if QWEN_VL_OCR_MODEL not in actual_model.lower():
                    logger.warning(f"   ⚠️ Model mismatch: requested '{QWEN_VL_OCR_MODEL}', got '{actual_model}'")
            
            # Extract result
            if response.choices and len(response.choices) > 0:
                result_text = response.choices[0].message.content
                logger.info(f"   ✅ Qwen-VL-OCR extracted {len(result_text)} characters")
                
                # Parse structured output
                clean_text, diagram_regions = self._parse_response(
                    result_text, 
                    image_width, 
                    image_height
                )
                
                return clean_text, diagram_regions
            else:
                raise RuntimeError(
                    "Qwen-VL-OCR returned empty response. OCR processing HALTED."
                )
                
        except RuntimeError:
            # Re-raise RuntimeError as-is (already formatted)
            raise
        except Exception as e:
            error_str = str(e).lower()
            
            # Detect quota/billing errors
            if any(keyword in error_str for keyword in ['quota', 'billing', 'limit', 'exceeded', 'insufficient', 'credit']):
                error_msg = (
                    f"QUOTA/BILLING ERROR: {e}. "
                    "The free-tier quota may have ended or billing limit exceeded. "
                    "OCR processing HALTED."
                )
                logger.error(f"   ❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            # Detect authentication errors
            if any(keyword in error_str for keyword in ['unauthorized', 'invalid key', 'authentication', '401', '403']):
                error_msg = (
                    f"AUTHENTICATION ERROR: {e}. "
                    "Check ALIBABA_API_KEY environment variable. "
                    "OCR processing HALTED."
                )
                logger.error(f"   ❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            # Detect timeout errors
            if any(keyword in error_str for keyword in ['timeout', 'timed out', 'connection']):
                error_msg = (
                    f"CONNECTION/TIMEOUT ERROR: {e}. "
                    "OCR processing HALTED."
                )
                logger.error(f"   ❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            # Detect generic error
            logger.error(f"   ❌ Qwen-VL-OCR error: {e}")
            raise RuntimeError(f"Qwen-VL-OCR failed: {e}. OCR processing HALTED.")
    
    def _strip_tags(self, text: str) -> str:
        """
        Strip HTML tags and markdown code blocks from the OCR response.
        
        Args:
            text: Raw OCR output
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        text = html.unescape(text)
            
        text = re.sub(r'```(?:[a-zA-Z0-9]+)?\s*', '', text)
        text = re.sub(r'```', '', text)
        
        text = re.sub(r'</(?:p|div|br|li|tr|h[1-6])\s*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<(?:p|div|br|li|tr|h[1-6])(?:\s+[^>]*)?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join([line for line in lines if line])
        
        return text.strip()
    
    def _get_ocr_prompt(self) -> str:
        """
        Get OCR prompt for structured text extraction.
        
        Instructs Qwen-VL-OCR to:
        - Extract all readable text as editable text
        - Convert equations to LaTeX
        - Mark diagram regions
        - Preserve document structure
        """
        return """You are a precise OCR system for handwritten notes.

TASK: Extract ALL text from this image as editable, structured content.

OUTPUT REQUIREMENTS:

1. TEXT EXTRACTION:
   - Extract ALL readable text exactly as written
   - Use "## " prefix for main headings/titles
   - Use "### " prefix for subheadings
   - Preserve paragraph breaks with blank lines
   - Keep bullet points as "- " and numbered lists

2. MATHEMATICAL EQUATIONS:
   - Convert ALL mathematical expressions to LaTeX format
   - Use $$...$$ for standalone equations (on their own line)
   - Use $...$ for inline math within text
   - Examples: $$E = mc^2$$, $\\frac{x}{y}$, $\\alpha + \\beta$

3. DIAGRAMS/CHARTS/FIGURES:
   - When you see a diagram, chart, graph, or drawing:
   - Insert: [[DIAGRAM:position=X%,description=brief description]]
   - X% is the vertical position (0-100) in the document
   - Continue transcribing text after the diagram marker

4. DO NOT:
   - Use HTML tags (e.g., <html>, <body>, <p>, <table>)
   - Wrap output in markdown code blocks (```)
   - Skip any text content
   - Convert text regions to descriptions
   - Apply any image processing
   - Summarize or paraphrase the content

OUTPUT FORMAT:
Return structured text with:
- Headings on their own lines with ## or ###
- LaTeX equations in $...$ or $$...$$
- [[DIAGRAM:...]] markers for visual elements
- Paragraphs separated by blank lines

Extract EVERYTHING as editable text."""
    
    def _parse_response(
        self,
        text: str,
        image_width: int = None,
        image_height: int = None
    ) -> Tuple[str, List[Dict]]:
        """
        Parse Qwen-VL-OCR response to extract text and diagram metadata.
        
        Args:
            text: Raw OCR output
            image_width: Image width for bbox calculation
            image_height: Image height for bbox calculation
            
        Returns:
            Tuple of (clean_text, diagram_regions)
        """
        import re
        
        # CRITICAL: Strip HTML tags and markdown code blocks first
        text = self._strip_tags(text)
        
        diagram_regions = []
        diagram_idx = 0
        
        # Pattern: [[DIAGRAM:position=X%,description=...]]
        pattern = r'\[\[DIAGRAM:position=(\d+)%(?:,description=([^\]]+))?\]\]'
        
        def replace_diagram(match):
            nonlocal diagram_idx
            position_pct = int(match.group(1))
            description = match.group(2) or "diagram"
            
            # Estimate bounding box if we have image dimensions
            if image_height and image_width:
                # Estimate vertical position
                y_start = int((position_pct / 100.0) * image_height * 0.85)
                y_end = min(y_start + int(image_height * 0.25), image_height)
            else:
                y_start = 0
                y_end = 0
            
            diagram_regions.append({
                'index': diagram_idx,
                'position_percent': position_pct,
                'description': description.strip(),
                'estimated_bbox': {
                    'x1': 0,
                    'y1': y_start,
                    'x2': image_width or 0,
                    'y2': y_end
                } if image_height else None
            })
            
            placeholder = f"[[DIAGRAM_{diagram_idx}]]"
            diagram_idx += 1
            return placeholder
        
        clean_text = re.sub(pattern, replace_diagram, text)
        
        # Also handle simpler [[DIAGRAM]] markers
        simple_pattern = r'\[\[DIAGRAM\]\]'
        
        def replace_simple_diagram(match):
            nonlocal diagram_idx
            diagram_regions.append({
                'index': diagram_idx,
                'position_percent': 50,
                'description': 'diagram',
                'estimated_bbox': None
            })
            placeholder = f"[[DIAGRAM_{diagram_idx}]]"
            diagram_idx += 1
            return placeholder
        
        clean_text = re.sub(simple_pattern, replace_simple_diagram, clean_text)
        
        # Clean up extra whitespace while preserving structure
        clean_text = clean_text.strip()
        
        return clean_text, diagram_regions


# Global instance
_qwen_vl_ocr = None


def get_qwen_vl_ocr() -> QwenVLOCR:
    """
    Get or create QwenVLOCR instance.
    
    Returns:
        QwenVLOCR instance
    """
    global _qwen_vl_ocr
    if _qwen_vl_ocr is None:
        _qwen_vl_ocr = QwenVLOCR()
    return _qwen_vl_ocr
