from typing import Dict, Optional
import logging
from pathlib import Path
from src.response_parser import EthicalAnalysis

class PDFGenerator:
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = templates_dir
        self.css_file = Path(templates_dir) / "styles.css"
        self._weasyprint_available = False
        
        try:
            from weasyprint import HTML, CSS
            self.HTML = HTML
            self.CSS = CSS
            self._weasyprint_available = True
        except (ImportError, OSError) as e:
            logging.warning(f"WeasyPrint not available: {str(e)}")
            logging.warning("PDF generation will be disabled")
    
    def generate_pdf(self, html_content: str, output_path: str) -> Optional[str]:
        """Generate PDF from HTML content"""
        if not self._weasyprint_available:
            logging.error("PDF generation is not available - WeasyPrint is not properly installed")
            return None
            
        try:
            from tempfile import NamedTemporaryFile
            
            # Create a temporary HTML file
            with NamedTemporaryFile(suffix='.html', mode='w', encoding='utf-8') as tmp:
                tmp.write(html_content)
                tmp.flush()
                
                # Create PDF
                self.HTML(tmp.name).write_pdf(
                    output_path,
                    stylesheets=[self.CSS(str(self.css_file))] if self.css_file.exists() else None
                )
            
            return output_path
        except Exception as e:
            logging.error(f"Error generating PDF: {str(e)}")
            return None 