from typing import Dict
from dataclasses import dataclass
from pathlib import Path

class SystemConfig:
    def __init__(self):
        self.model = "gpt-4"  # Default model
        self.system_prompt = """You are an AI assistant specializing in medical ethics. 
        Provide detailed, structured analysis of ethical considerations in healthcare."""
        self.max_tokens = 1000
        self.temperature = 0.7
        
        # PubMed settings
        self.pubmed_email = "medical.ethics.assistant@example.com"  # Default email
        self.pubmed_api_key = None  # Optional
        self.max_results = 5
        
        # Database settings
        self.db_path = "ethics.db"
        
        # Response settings
        self.include_references = True
        self.max_follow_up_questions = 3
        
    @property
    def model_config(self):
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

class EthicsCategories:
    CATEGORIES = [
        "informed_consent",
        "privacy_confidentiality",
        "risk_benefit",
        "vulnerable_populations",
        "data_security",
        "conflict_of_interest",
        "research_integrity"
    ]
    
    PRINCIPLES = {
        "autonomy": "Respect for individual autonomy and self-determination",
        "beneficence": "Promoting well-being and preventing harm",
        "justice": "Fair distribution of benefits and risks",
        "non_maleficence": "Avoiding harm to participants"
    } 