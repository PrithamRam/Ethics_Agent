from typing import Dict, List
from jinja2 import Environment, FileSystemLoader
import os
from dataclasses import asdict
from src.response_parser import EthicalAnalysis
from jinja2 import Template
import logging

logger = logging.getLogger(__name__)

class TemplateManager:
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = templates_dir
        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=True
        )
    
    def generate_response_html(self, query: str, analysis: EthicalAnalysis, references: List[Dict], total_results: int = None) -> str:
        """Generate HTML response"""
        try:
            logger.info("Generating HTML response")
            logger.debug(f"Analysis: {analysis}")
            logger.debug(f"Number of references: {len(references)}")
            logger.debug(f"Total PubMed results: {total_results}")
            
            template = self.env.get_template('ethics_response.html')
            return template.render(
                query=query,
                analysis=analysis,
                references=references,
                total_results=total_results
            )
        except Exception as e:
            logger.error(f"Error generating HTML response: {str(e)}", exc_info=True)
            error_template = self.env.get_template('error.html')
            return error_template.render(error=str(e))
    
    def generate_error_html(self, error_message: str) -> str:
        """Generate error response HTML"""
        template = self.env.get_template("error.html")
        return template.render(error=error_message)

    def generate_follow_up_html(self, response: str, references: List[Dict] = None) -> str:
        """Generate HTML for follow-up response"""
        try:
            template = self.env.get_template('follow_up_response.html')
            return template.render(
                response=response,
                references=references or []
            )
        except Exception as e:
            logger.error(f"Error generating follow-up HTML: {str(e)}")
            return f"<div class='error'>Error generating response: {str(e)}</div>" 