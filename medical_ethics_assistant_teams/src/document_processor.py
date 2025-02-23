from typing import Dict, List
import re
from dataclasses import dataclass
from src.config import EthicsCategories

@dataclass
class Reference:
    pubmed_id: str
    title: str
    abstract: str
    keywords: List[str]
    ethical_considerations: List[str]

class DocumentProcessor:
    def __init__(self):
        self.categories = EthicsCategories.CATEGORIES
        self.principles = EthicsCategories.PRINCIPLES
    
    async def process_file(self, file_path: str) -> Dict:
        """Process the abstract-ethics.txt file into structured data"""
        with open(file_path, 'r') as file:
            content = file.read()
        
        structured_data = self._parse_content(content)
        return structured_data
    
    def _parse_content(self, content: str) -> Dict:
        """Parse content into structured format"""
        references = []
        current_reference = None
        
        # Split content into sections
        sections = re.split(r'\n(?=PMID:)', content)
        
        for section in sections:
            if section.strip():
                reference = self._parse_reference(section)
                if reference:
                    references.append(reference)
        
        return {
            "references": references,
            "principles": self._extract_principles(content),
            "categories": self._extract_categories(content)
        }
    
    def _parse_reference(self, section: str) -> Reference:
        """Parse individual reference section"""
        pmid_match = re.search(r'PMID:\s*(\d+)', section)
        if not pmid_match:
            return None
            
        pubmed_id = pmid_match.group(1)
        title = self._extract_title(section)
        abstract = self._extract_abstract(section)
        keywords = self._extract_keywords(section)
        considerations = self._extract_ethical_considerations(section)
        
        return Reference(
            pubmed_id=pubmed_id,
            title=title,
            abstract=abstract,
            keywords=keywords,
            ethical_considerations=considerations
        )

    def _extract_title(self, section: str) -> str:
        """Extract title from the reference section"""
        # Look for title after PMID until the next section
        title_match = re.search(r'PMID:\s*\d+\s*(.+?)(?=\nAbstract:|$)', section, re.DOTALL)
        if title_match:
            return title_match.group(1).strip()
        return ""

    def _extract_abstract(self, section: str) -> str:
        """Extract abstract from the reference section"""
        abstract_match = re.search(r'Abstract:\s*(.+?)(?=\nKeywords:|$)', section, re.DOTALL)
        if abstract_match:
            return abstract_match.group(1).strip()
        return ""

    def _extract_keywords(self, section: str) -> List[str]:
        """Extract keywords from the reference section"""
        keywords_match = re.search(r'Keywords:\s*(.+?)(?=\nEthical Considerations:|$)', section, re.DOTALL)
        if keywords_match:
            keywords_text = keywords_match.group(1)
            # Split by commas and clean up each keyword
            return [k.strip() for k in keywords_text.split(',') if k.strip()]
        return []

    def _extract_ethical_considerations(self, section: str) -> List[str]:
        """Extract ethical considerations from the reference section"""
        considerations_match = re.search(r'Ethical Considerations:\s*(.+?)(?=\nPMID:|$)', section, re.DOTALL)
        if considerations_match:
            considerations_text = considerations_match.group(1)
            # Split by bullet points or numbered lists and clean
            considerations = re.split(r'\n\s*[-•]\s*|\n\s*\d+\.\s*', considerations_text)
            return [c.strip() for c in considerations if c.strip()]
        return []

    def _extract_principles(self, content: str) -> Dict:
        """Extract ethical principles and their occurrences"""
        principles_dict = {}
        for principle, description in self.principles.items():
            # Count occurrences and collect context
            matches = re.finditer(rf'\b{principle}\b', content, re.IGNORECASE)
            contexts = []
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end].strip()
                contexts.append(context)
            
            principles_dict[principle] = {
                'description': description,
                'occurrences': len(contexts),
                'contexts': contexts
            }
        return principles_dict

    def _extract_categories(self, content: str) -> Dict:
        """Extract ethical categories and their relevance"""
        categories_dict = {}
        for category in self.categories:
            # Replace underscores with spaces for searching
            search_term = category.replace('_', ' ')
            matches = re.finditer(rf'\b{search_term}\b', content, re.IGNORECASE)
            contexts = []
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end].strip()
                contexts.append(context)
            
            categories_dict[category] = {
                'occurrences': len(contexts),
                'contexts': contexts
            }
        return categories_dict 