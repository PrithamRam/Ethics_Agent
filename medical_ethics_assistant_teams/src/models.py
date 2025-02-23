from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Reference:
    pubmed_id: str
    title: str
    abstract: str
    keywords: List[str]
    ethical_considerations: List[str]
    authors: Optional[List[str]] = None
    publication_date: Optional[str] = None 