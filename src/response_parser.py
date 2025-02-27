from typing import Dict, List, Any
import re
from dataclasses import dataclass
import logging
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

logger = logging.getLogger(__name__)

@dataclass
class EthicalAnalysis:
    summary: str
    recommendations: List[str]
    concerns: List[str]
    mitigation_strategies: List[str]
    follow_up_questions: List[str]

class ResponseParser:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=self.api_key)

    def parse_gpt_response(self, response: str) -> Dict[str, Any]:
        """Parse GPT response into structured format"""
        logger.info("Starting to parse GPT response...")
        
        sections = {
            'summary': '',
            'recommendations': [],
            'concerns': [],
            'mitigation_strategies': [],
            'follow_up_questions': []
        }
        
        try:
            # Split into sections
            current_section = None
            current_text = []
            
            for line in response.split('\n'):
                line = line.strip()
                if not line or line.startswith('Question:'):
                    continue
                    
                # Log each line for debugging
                logger.debug(f"Processing line: {line}")
                
                # Check for section headers (exact matches)
                if line == 'SUMMARY':
                    current_section = 'summary'
                    continue
                elif line == 'RECOMMENDATIONS':
                    current_section = 'recommendations'
                    continue
                elif line == 'ETHICAL CONCERNS':
                    current_section = 'concerns'
                    continue
                elif line == 'MITIGATION STRATEGIES':
                    current_section = 'mitigation_strategies'
                    continue
                elif line == 'FOLLOW-UP QUESTIONS':
                    current_section = 'follow_up_questions'
                    continue
                
                # Add content to current section
                if current_section:
                    if current_section == 'summary':
                        current_text.append(line)
                    elif line.startswith('- '):
                        sections[current_section].append(line[2:].strip())
            
            # Save summary text
            if current_text:
                sections['summary'] = ' '.join(current_text)
            
            logger.info(f"Final parsed sections: {json.dumps(sections, indent=2)}")
            return sections
            
        except Exception as e:
            logger.error(f"Error parsing GPT response: {str(e)}", exc_info=True)
            return sections
    
    def _split_into_sections(self, response: str) -> Dict[str, List[str]]:
        """Split response into different sections"""
        sections = {
            'Summary': '',
            'Recommendations': [],
            'Concerns': [],
            'Mitigation Strategies': [],
            'Follow-up Questions': []
        }
        
        current_section = None
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            lower_line = line.lower()
            if 'summary:' in lower_line:
                current_section = 'Summary'
                continue
            elif any(x in lower_line for x in ['recommendation', 'key recommendation']):
                current_section = 'Recommendations'
                continue
            elif 'concern' in lower_line:
                current_section = 'Concerns'
                continue
            elif 'mitigation' in lower_line:
                current_section = 'Mitigation Strategies'
                continue
            elif 'follow-up' in lower_line or 'follow up' in lower_line:
                current_section = 'Follow-up Questions'
                continue
                
            if current_section:
                # Handle numbered or bulleted items
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                    line = line.lstrip('123456789.-• ')
                
                if current_section == 'Summary':
                    sections[current_section] = line
                else:
                    sections[current_section].append(line)
        
        return sections
    
    def _extract_summary(self, sections: Dict[str, List[str]]) -> str:
        """Extract and clean summary section"""
        return sections['Summary']
    
    def _extract_recommendations(self, sections: Dict[str, List[str]]) -> List[str]:
        """Extract recommendations as list"""
        return sections['Recommendations']
    
    def _extract_concerns(self, sections: Dict[str, List[str]]) -> List[str]:
        """Extract concerns as list"""
        return sections['Concerns']
    
    def _extract_mitigations(self, sections: Dict[str, List[str]]) -> List[str]:
        """Extract mitigation strategies as list"""
        return sections['Mitigation Strategies']
    
    def _extract_citations(self, sections: Dict[str, List[str]]) -> List[str]:
        """Extract citations as list"""
        return sections.get('References', [])
    
    def _extract_follow_up_questions(self, sections: Dict[str, List[str]]) -> List[str]:
        """Extract follow-up questions as list"""
        return sections.get('Follow-up Questions', [])
    
    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text"""
        points = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                # Remove bullet point markers and clean
                clean_line = re.sub(r'^[•\-*\d.]+\s*', '', line)
                points.append(clean_line)
        return points

    def _analyze_ethical_principles(self, text: str) -> Dict[str, List[str]]:
        """Analyze text for ethical principles"""
        principles = {
            'autonomy': [],
            'beneficence': [],
            'non_maleficence': [],
            'justice': []
        }
        
        # Look for principle-related content
        for principle in principles.keys():
            matches = re.finditer(
                rf'\b{principle}\b.+?[.!?]',
                text,
                re.IGNORECASE | re.DOTALL
            )
            principles[principle] = [m.group(0).strip() for m in matches]
        
        return principles

    def _extract_key_stakeholders(self, text: str) -> List[str]:
        """Extract key stakeholders mentioned in the text"""
        stakeholder_patterns = [
            r'patients?',
            r'researchers?',
            r'participants?',
            r'physicians?',
            r'healthcare providers?',
            r'institutions?'
        ]
        
        stakeholders = set()
        for pattern in stakeholder_patterns:
            matches = re.finditer(rf'\b{pattern}\b', text, re.IGNORECASE)
            stakeholders.update(m.group(0).lower() for m in matches)
        
        return list(stakeholders)

    def _extract_risk_levels(self, text: str) -> Dict[str, List[str]]:
        """Extract risk levels and associated concerns"""
        risk_levels = {
            'high': [],
            'moderate': [],
            'low': []
        }
        
        for level in risk_levels.keys():
            pattern = rf'\b{level}\s+risk\b.+?[.!?]'
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            risk_levels[level] = [m.group(0).strip() for m in matches]
        
        return risk_levels

    def generate_response(self, question: str, papers: List[Dict]) -> Dict:
        """Generate two-part response: paper analysis and general knowledge"""
        try:
            # First response: Based on papers
            papers_response = self._generate_papers_response(question, papers)
            
            # Second response: General knowledge
            general_response = self._generate_general_response(question)
            
            # Return structured response
            return {
                "literature_analysis": papers_response,
                "general_analysis": general_response,
                "papers": papers
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "literature_analysis": f"Error analyzing papers: {str(e)}",
                "general_analysis": "",
                "papers": papers
            }

    def _generate_general_response(self, question: str) -> str:
        """Generate response based on general knowledge"""
        prompt = f"""As an expert in medical ethics, provide an analysis of this question 
        WITHOUT referring to any specific research papers: {question}

        Please structure your response with:
        1. Key ethical principles involved
        2. Potential benefits and risks
        3. Practical considerations
        4. Recommendations for ethical implementation
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in medical ethics. Provide analysis based on established ethical principles and general knowledge."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content

    def _generate_papers_response(self, question: str, papers: List[Dict]) -> str:
        """Generate literature-based analysis with strict paper references"""
        try:
            if not papers:
                return """No directly relevant papers were found in the literature search. 
                A broader search or consultation with medical ethics experts is recommended."""
            
            # Format papers into context with limit
            context = self._format_papers_context(papers)
            
            prompt = f"""Analyze these specific papers in relation to the case. 
            ONLY use information explicitly stated in these papers.
            If there isn't clear guidance from these papers, say so.

Case:
{question}

Available Papers:
{context}

Please provide:
1. Evidence Summary:
   - What specific guidance exists in these papers
   - ONLY cite information actually present in the papers
   - Clearly indicate if certain aspects aren't addressed

2. Strength of Evidence:
   - How directly relevant are these papers
   - Note any limitations in applying their guidance
   - Identify gaps in the available evidence

3. Synthesis:
   - Only combine guidance that's explicitly in these papers
   - Don't extrapolate beyond what's stated
   - Clearly indicate when more evidence would be needed

Important: Do NOT make assumptions or add information not found in these specific papers."""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a medical ethics expert. Be precise and only reference the provided papers. Do not make assumptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more conservative responses
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating papers response: {str(e)}")
            return "Error analyzing papers. Please review the papers directly."

    def _format_papers_context(self, papers: List[Dict]) -> str:
        """Format papers with clear identification"""
        context = []
        total_chars = 0
        char_limit = 6000
        
        for i, paper in enumerate(papers, 1):
            summary = f"""Paper {i}:
Title: {paper.get('title', 'No title')}
Authors: {', '.join(f"{a.get('last_name', '')}" for a in paper.get('authors', []))}
Year: {paper.get('year', 'Year not specified')}
Abstract: {paper.get('abstract', 'No abstract available')}
Key Points: {', '.join(paper.get('ethical_considerations', ['None specified']))}
PMID: {paper.get('pubmed_id', 'No ID')}
---"""
            
            if total_chars + len(summary) > char_limit:
                break
            context.append(summary)
            total_chars += len(summary)
        
        if not context:
            return "No papers available for analysis."
        
        return "\n\n".join(context)

    def _assess_relevance(self, paper: Dict) -> str:
        """Assess conceptual relevance rather than keyword matching"""
        try:
            # Create a summary of the paper's ethical content
            content = f"""
            Title: {paper.get('title', '')}
            Abstract: {paper.get('abstract', '')}
            Ethical Points: {', '.join(paper.get('ethical_considerations', []))}
            """
            
            # Analyze conceptual relevance
            prompt = f"""Rate this paper's relevance to ethical principles of:
            - Stakeholder relationships
            - Core ethical principles
            - Type of ethical dilemma
            
            Paper content:
            {content}
            
            Return only: "direct", "conceptual", or "general"
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at identifying ethical relevance. Be concise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            return response.choices[0].message.content.strip().lower()
            
        except Exception as e:
            logger.error(f"Error assessing relevance: {str(e)}")
            return "general"

    def _create_detailed_summary(self, paper: Dict) -> str:
        """Create detailed summary focusing on ethical principles"""
        return f"""Paper: {paper['title']}
Key Ethical Principles: {', '.join(paper.get('ethical_considerations', ['None specified']))}
Relevant Guidance: {self._extract_main_guidance(paper)}
Application: {self._suggest_principle_application(paper)}
---"""

    def _suggest_principle_application(self, paper: Dict) -> str:
        """Suggest how paper's principles might apply to other contexts"""
        abstract = paper.get('abstract', '')
        if not abstract:
            return "No application specified"
        
        # Look for transferable principles
        principle_indicators = [
            "principle", "framework", "approach", "consideration",
            "balance", "value", "right", "obligation"
        ]
        
        relevant_parts = []
        for sentence in abstract.split('. '):
            if any(indicator in sentence.lower() for indicator in principle_indicators):
                relevant_parts.append(sentence)
        
        if relevant_parts:
            return '. '.join(relevant_parts)
        return "No explicit principles found"

    def _extract_main_guidance(self, paper: Dict) -> str:
        """Extract main ethical guidance from paper"""
        abstract = paper.get('abstract', '')
        if not abstract:
            return "No guidance specified"
        
        # Look for recommendation/guidance sections
        guidance_indicators = [
            "recommend", "suggest", "conclude", "propose",
            "should", "must", "guideline", "principle"
        ]
        
        relevant_sentences = []
        for sentence in abstract.split('. '):
            if any(indicator in sentence.lower() for indicator in guidance_indicators):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            return '. '.join(relevant_sentences)
        return "No explicit guidance found" 