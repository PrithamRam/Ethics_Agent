from typing import Dict, List
import re
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EthicalAnalysis:
    summary: str
    recommendations: List[str]
    concerns: List[str]
    mitigation_strategies: List[str]
    follow_up_questions: List[str]

class ResponseParser:
    def parse_gpt_response(self, response_text: str) -> dict:
        """Parse GPT response into structured format"""
        try:
            # Default structure
            analysis = {
                'summary': '',
                'recommendations': [],
                'follow_up_questions': [],
                'concerns': [],
                'mitigation_strategies': []
            }

            # If response is just text, use it as summary
            if not any(marker in response_text.lower() for marker in ['summary:', 'recommendations:', 'follow-up questions:']):
                analysis['summary'] = response_text.strip()
                return analysis

            # Split response into sections
            current_section = 'summary'
            current_content = []

            for line in response_text.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Check for section headers
                lower_line = line.lower()
                if 'summary:' in lower_line:
                    current_section = 'summary'
                    continue
                elif any(x in lower_line for x in ['recommendations:', 'key points:', 'recommendations and points:']):
                    current_section = 'recommendations'
                    continue
                elif any(x in lower_line for x in ['follow-up questions:', 'additional questions:', 'further questions:']):
                    current_section = 'follow_up_questions'
                    continue
                elif 'concerns:' in lower_line:
                    current_section = 'concerns'
                    continue
                elif 'mitigation strategies:' in lower_line:
                    current_section = 'mitigation_strategies'
                    continue

                # Remove list markers and clean the line
                cleaned_line = line.lstrip('•-*1234567890. ')
                if cleaned_line:
                    if current_section == 'summary':
                        if analysis['summary']:
                            analysis['summary'] += ' ' + cleaned_line
                        else:
                            analysis['summary'] = cleaned_line
                    else:
                        analysis[current_section].append(cleaned_line)

            # Ensure we have at least a summary if nothing was parsed
            if not any(analysis.values()):
                analysis['summary'] = response_text.strip()

            return analysis

        except Exception as e:
            logger.error(f"Error parsing GPT response: {str(e)}")
            # Return raw response as summary if parsing fails
            return {
                'summary': response_text.strip(),
                'recommendations': [],
                'follow_up_questions': [],
                'concerns': [],
                'mitigation_strategies': []
            }
    
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