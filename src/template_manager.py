from typing import Dict, List
from jinja2 import Environment, FileSystemLoader
import logging

logger = logging.getLogger(__name__)

class TemplateManager:
    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader('templates'),
            autoescape=True
        )
    
    def generate_response_html(self, query: str, analysis: Dict, references: List[Dict], total_results: int) -> str:
        """Generate HTML response using the ethics_response template"""
        try:
            template = self.env.get_template('ethics_response.html')
            return template.render(
                query=query,
                analysis=analysis,
                references=references,
                total_results=total_results
            )
        except Exception as e:
            logger.error(f"Error generating response HTML: {str(e)}")
            return self.generate_error_html(str(e))
    
    def generate_error_html(self, error_message: str) -> str:
        """Generate HTML for error messages"""
        try:
            template = self.env.get_template('error.html')
            return template.render(error=error_message, back_url='/')
        except Exception as e:
            logger.error(f"Error generating error HTML: {str(e)}")
            return f"<div class='error'>Error: {error_message}</div>" 