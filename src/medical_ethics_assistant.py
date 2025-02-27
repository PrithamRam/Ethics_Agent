from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from openai import OpenAI
from src.pubmed_handler import PubMedHandler
from src.ethics_db import EthicsDB
from src.document_processor import DocumentProcessor
from src.config import SystemConfig
import json
import logging
from src.response_parser import ResponseParser, EthicalAnalysis
from src.template_manager import TemplateManager
from datetime import datetime

logger = logging.getLogger(__name__)  # Add logger

class MedicalEthicsAssistant:
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        load_dotenv()
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.openai_key)
        
        # Initialize components without passing self
        self.pubmed_handler = PubMedHandler()
        self.ethics_db = EthicsDB()
        self.doc_processor = DocumentProcessor()
        self.response_parser = ResponseParser()
        self.template_manager = TemplateManager()
        self.conversation_history = []
        self.current_query_context = None
        self.last_response = None

    @classmethod
    async def create(cls, config: SystemConfig = None) -> 'MedicalEthicsAssistant':
        """Factory method to create and initialize the assistant"""
        # Create instance without passing config to __init__
        instance = cls()
        # Set config after creation if provided
        if config:
            instance.config = config
        # Initialize database connection
        await instance.ethics_db.get_connection()
        return instance
    
    async def initialize_knowledge_base(self, abstract_ethics_path: str):
        """Initialize the knowledge base from the abstract-ethics.txt file"""
        try:
            processed_data = await self.doc_processor.process_file(abstract_ethics_path)
            await self.ethics_db.populate_database(processed_data)
            logging.info("Knowledge base initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing knowledge base: {str(e)}")
            raise
    
    async def _get_gpt_response(self, prompt: str, context: str = None) -> str:
        """Get response from GPT model"""
        try:
            messages = []
            
            # Add system message with context-specific instructions
            if "Format the response as a strict JSON object" in prompt:
                # For search terms extraction
                messages.append({
                    "role": "system",
                    "content": """You are a medical ethics expert that extracts key terms from cases.
                    When asked to return JSON:
                    1. Return ONLY a valid JSON object with no additional text
                    2. Do not use markdown formatting or code blocks
                    3. Each category must contain an array of strings
                    4. Include at least 2-3 terms in each relevant category
                    5. Medical terms should be specific (e.g., "advanced dementia" not just "dementia")
                    6. Ethical terms should be precise (e.g., "patient autonomy" not just "autonomy")
                    7. Example format:
                    {
                        "medical_conditions": ["advanced dementia", "acute kidney failure"],
                        "treatments": ["dialysis", "palliative care"],
                        "ethical_principles": ["patient autonomy", "beneficence"],
                        "care_settings": ["intensive care unit", "long-term care facility"],
                        "key_issues": ["guardian reluctance", "resource allocation"]
                    }"""
                })
            else:
                # For other analysis tasks
                messages.append({
                    "role": "system",
                    "content": """You are an AI assistant specializing in medical ethics. Your responses must:

1. Use EXACTLY the section headers provided in the prompt
2. Format all lists with "- " bullet points (no numbers)
3. Provide detailed, substantive content for each section
4. Keep the summary section as a paragraph without bullets
5. Include at least 3 bullet points for each list section
6. Focus on practical, actionable recommendations
7. Support points with ethical principles and reasoning
8. Avoid section numbers in the content itself

Format sections exactly as:
SUMMARY
(paragraph)

RECOMMENDATIONS
- First point
- Second point
- Third point

And so on for other sections."""
                })
            
            # Add context if provided
            if context:
                messages.append({
                    "role": "assistant",
                    "content": f"Previous context:\n{context}"
                })
            
            # Add user prompt
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            logger.info(f"Sending request to OpenAI API with {len(messages)} messages")
            
            # Get GPT response using new client
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=2000  # Increased token limit for more detailed responses
            )
            
            if not response.choices:
                raise ValueError("No response choices returned from OpenAI")
            
            content = response.choices[0].message.content
            logger.info(f"Received response of length {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Error in GPT response: {str(e)}", exc_info=True)
            if "api_key" in str(e).lower():
                raise ValueError("OpenAI API key is invalid or not set. Please check your .env file.")
            raise
    
    async def _analyze_papers_for_case(self, papers: List[Dict], question: str) -> str:
        """Analyze papers for their applicability to the current case"""
        try:
            # If no papers found, return early
            if not papers:
                return "No relevant papers found in the database."
            
            # Format papers for analysis
            papers_to_analyze = []
            for paper in papers:
                # Convert author format if needed
                if 'authors' in paper:
                    if isinstance(paper['authors'], list):
                        if all(isinstance(a, str) for a in paper['authors']):
                            # Convert string authors to dict format
                            paper['authors'] = [{'last_name': a} for a in paper['authors']]
                        # else assume it's already in correct format
                    else:
                        paper['authors'] = [{'last_name': 'Unknown'}]
                else:
                    paper['authors'] = [{'last_name': 'Unknown'}]
                
                papers_to_analyze.append(paper)
            
            prompt = f"""Analyze these medical ethics papers specifically for this case:

Case:
{question}

Papers to analyze:
{self._format_papers_for_prompt(papers_to_analyze)}

Please provide:
1. EVIDENCE-BASED RECOMMENDATIONS
   - Provide 3-4 specific recommendations
   - Each recommendation must cite a specific paper
   - Explain how each recommendation applies to this case

2. SYNTHESIS
   - Integrate the perspectives from all papers
   - Summarize the key ethical principles identified
   - Explain how these principles apply to this case

Use EXACTLY these section headers and format recommendations with paper citations.
"""

            response = await self._get_gpt_response(prompt)
            return response

        except Exception as e:
            logger.error(f"Error analyzing papers: {str(e)}", exc_info=True)
            return "Error analyzing literature"

    def _format_papers_for_prompt(self, papers: List[Dict]) -> str:
        """Format papers for GPT analysis"""
        formatted = []
        for paper in papers:
            # Format authors safely
            authors = paper.get('authors', [])
            if isinstance(authors, list):
                author_names = ', '.join(a.get('last_name', '') for a in authors)
            else:
                author_names = 'Unknown'

            paper_text = f"""
Title: {paper.get('title', 'Untitled')}
Authors: {author_names}
Year: {paper.get('year', 'N/A')}
Journal: {paper.get('journal', 'N/A')}
Abstract: {paper.get('abstract', 'No abstract available')}
Ethical Considerations: {', '.join(paper.get('ethical_considerations', []))}
"""
            formatted.append(paper_text)
        
        return "\n---\n".join(formatted)

    async def get_ethical_guidance(self, query: str) -> Dict[str, Any]:
        """Get ethical guidance for a query"""
        try:
            logger.info("Getting initial AI analysis...")
            ai_analysis = await self._get_initial_analysis(query)
            logger.info("Got AI analysis")

            logger.info("Getting relevant papers...")
            papers = await self.get_relevant_papers(query, ai_analysis)
            logger.info(f"Found {len(papers['papers'])} relevant papers")

            logger.info("Analyzing papers...")
            lit_analysis = await self._analyze_papers_for_case(papers['papers'], query)
            logger.info("Generated literature analysis")

            response = {
                "ai_analysis": ai_analysis,
                "literature_analysis": lit_analysis,
                "relevant_papers": papers['papers'],
                "search_terms": papers['search_terms']
            }

            logger.info(f"Final response: {json.dumps(response, indent=2)}")
            return response

        except Exception as e:
            logger.error(f"Error getting ethical guidance: {str(e)}", exc_info=True)
            raise
    
    async def _analyze_literature(self, refs: List[Dict]) -> str:
        """Analyze the literature evidence"""
        if not refs:
            return "No relevant literature found."
            
        # Build context from papers
        papers_context = []
        for ref in refs:
            try:
                # Fetch paper content from PubMed
                paper_details = await self.pubmed_handler.get_paper_details(ref['pubmed_id'])
                if paper_details and paper_details.get('abstract'):
                    papers_context.append(f"""
                    Paper: {ref['title']}
                    Authors: {ref['authors']}
                    PMID: {ref['pubmed_id']}
                    
                    Abstract:
                    {paper_details['abstract']}
                    
                    Key Points:
                    {paper_details.get('keywords', [])}
                    """)
            except Exception as e:
                logger.error(f"Error fetching paper for PMID {ref['pubmed_id']}: {str(e)}")

        if not papers_context:
            return "Could not retrieve detailed content for the relevant papers."

        # Get analysis of literature
        return await self._get_gpt_response(
            f"""Based on these research papers, provide a structured analysis:
            
            Papers:
            {papers_context}
            
            Please analyze the evidence from these papers and provide:
            1. Summary of Literature Evidence
            2. Key Findings from Papers
            3. Research-based Recommendations
            4. Gaps in Current Research
            """
        )
    
    def _merge_references(self, local_refs: List[Dict], pubmed_refs: List[Dict]) -> List[Dict]:
        """Merge and deduplicate references from local DB and PubMed"""
        seen_pmids = set()
        merged_refs = []
        
        # Add local references first (they have ethical considerations)
        for ref in local_refs:
            if ref['pubmed_id'] not in seen_pmids:
                seen_pmids.add(ref['pubmed_id'])
                merged_refs.append(ref)
        
        # Add PubMed references that aren't already included
        for ref in pubmed_refs:
            if ref['pubmed_id'] not in seen_pmids:
                seen_pmids.add(ref['pubmed_id'])
                merged_refs.append(ref)
        
        return merged_refs
    
    def _prepare_context(self, refs: List[Dict], pubmed_context: List[Dict]) -> str:
        """Prepare context for GPT prompt"""
        context_parts = []
        
        # Add reference information
        for ref in refs:
            context_parts.append(f"Reference {ref['pubmed_id']}:")
            context_parts.append(f"Title: {ref['title']}")
            if 'ethical_considerations' in ref and ref['ethical_considerations']:
                context_parts.append(f"Ethical Considerations:")
                for consideration in ref['ethical_considerations']:
                    context_parts.append(f"- {consideration}")
            elif 'abstract' in ref:
                context_parts.append(f"Abstract: {ref['abstract']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    async def _generate_gpt_response(self, query: str, context: str) -> str:
        """Generate response using GPT"""
        prompt = self._create_prompt(query, context)
        
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an ethical advisor for medical research."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=self.config.MAX_TOKENS
        )
        
        return response.choices[0].message.content
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for GPT"""
        return f"""
        Based on the following context and references, provide ethical guidance for this query:
        
        Query: {query}
        
        Context:
        {context}
        
        Please provide:
        1. A comprehensive ethical analysis
        2. Specific recommendations
        3. Potential concerns and mitigation strategies
        4. References to specific papers and guidelines
        """
    
    def _structure_response(self, response: str, refs: List[Dict]) -> Dict:
        """Structure the GPT response"""
        # Implementation for response structuring
        return {
            'analysis': response,
            'recommendations': [],  # Extract recommendations
            'concerns': [],  # Extract concerns
            'citations': []  # Extract citations
        }
    
    async def _generate_follow_up_questions(self, query: str, response: str) -> List[str]:
        """Generate relevant follow-up questions"""
        prompt = f"""
        Based on this ethical query and response, what are 3-5 important follow-up questions to consider?
        
        Query: {query}
        Response: {response}
        """
        
        follow_up_response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Generate follow-up questions for ethical consideration."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        # Parse and clean up the follow-up questions
        questions = follow_up_response.choices[0].message.content.split('\n')
        return [q.strip('1234567890. -') for q in questions if q.strip()]
    
    def _format_references(self, refs: List[Dict], pubmed_context: List[Dict]) -> List[Dict]:
        """Format references for the response"""
        formatted_refs = []
        for ref in refs:
            pubmed_info = next((p for p in pubmed_context if p['pubmed_id'] == ref['pubmed_id']), None)
            if pubmed_info:
                formatted_refs.append({
                    'pubmed_id': ref['pubmed_id'],
                    'title': pubmed_info['title'],
                    'authors': pubmed_info['authors'],
                    'publication_date': pubmed_info.get('publication_date', ''),
                    'relevance': 'Primary source' if ref.get('ethical_considerations') else 'Supporting reference'
                })
        return formatted_refs
    
    async def get_follow_up_response(self, question: str) -> dict:
        """Get response to a follow-up question"""
        try:
            # Get the original query context and response
            original_query = self.current_query_context['query'] if self.current_query_context else ""
            previous_response = self.last_response if self.last_response else ""
            
            # Get AI response with full conversation context
            response = await self._get_gpt_response(
                f"""Based on the previous conversation:

                Original Question: {original_query}

                Previous Response: {previous_response}

                Follow-up Question: {question}

                Please provide a structured response that builds on the previous discussion and addresses the follow-up question.
                Format your response with:
                1. Summary
                2. Key Points (specifically addressing the follow-up)
                3. Additional Follow-up Questions

                Keep your response focused on how this follow-up relates to the original ethical discussion.
                """
            )
            
            # Parse the response
            analysis = self.response_parser.parse_gpt_response(response)
            
            # Store this response for future context
            self.last_response = response
            
            # Get relevant references (prioritizing ones from original query)
            search_query = f"{question} AND {original_query} AND (ethics OR ethical OR guidelines)"
            pmids, total_results = await self.pubmed_handler.search_pubmed(search_query)
            references = await self.pubmed_handler.fetch_papers(pmids)
            
            return {
                'analysis': analysis,
                'references': references,
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"Error handling follow-up question: {str(e)}")
            raise
    
    def _format_conversation_history(self) -> str:
        """Format conversation history for context"""
        if not self.conversation_history:
            return ""
        
        history_parts = []
        for i, entry in enumerate(self.conversation_history[-3:]):  # Last 3 exchanges
            history_parts.append(f"Q{i+1}: {entry['question']}")
            history_parts.append(f"A{i+1}: {entry['response']}\n")
        
        return "\n".join(history_parts)
    
    async def initialize_database(self):
        """Initialize the database with key papers"""
        try:
            # Initialize with ventilator ethics paper
            await self.ethics_db.initialize_with_paper()
            
            # Get paper details from PubMed
            papers = await self.pubmed_handler.fetch_papers(['32381261'])
            if papers:
                paper = papers[0]
                # Update paper with full details
                await self.ethics_db._insert_reference(
                    await self.ethics_db.get_connection(),
                    Reference(
                        pubmed_id=paper['pubmed_id'],
                        title=paper['title'],
                        abstract=paper['abstract'],
                        keywords=['COVID-19', 'ventilators', 'ethics', 'resource allocation'],
                        ethical_considerations=[
                            'Fair allocation of scarce resources',
                            'Legal framework for rationing decisions',
                            'Ethical principles in triage',
                            'Protection of healthcare workers',
                            'Documentation requirements'
                        ]
                    )
                )
                logger.info(f"Initialized database with paper: {paper['title']}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            return False

    async def _extract_search_concepts(self, ai_analysis: Dict) -> Dict[str, float]:
        """Extract search concepts and their importance weights from AI analysis"""
        try:
            # Create prompt using the AI analysis
            prompt = f"""Extract key concepts and their importance weights from this AI analysis:

            Analysis: {ai_analysis['summary']}
            
            Return a Python dictionary with concepts and weights in this exact format:
            {{
                'ethical_principles': {{'autonomy': 0.9, 'beneficence': 0.8}},
                'medical_terms': {{'dementia': 0.9, 'kidney_failure': 0.8}},
                'stakeholder_roles': {{'guardian': 0.9, 'caregiver': 0.8}},
                'care_settings': {{'care_facility': 0.8, 'hospital': 0.7}}
            }}

            Use only lowercase words, underscores for spaces, and weights between 0.0-1.0.
            Include only the most relevant terms from the analysis.
            """
            
            response = await self._get_gpt_response(prompt)
            # Clean the response string and safely evaluate it
            cleaned_response = response.strip().replace('\n', '').replace(' ', '')
            concepts = eval(cleaned_response)
            return concepts
        except Exception as e:
            logger.error(f"Error extracting search concepts: {str(e)}")
            # Return default concepts if extraction fails
            return {
                'ethical_principles': {'autonomy': 0.9, 'beneficence': 0.8},
                'medical_terms': {'dementia': 0.8},
                'stakeholder_roles': {'guardian': 0.9},
                'care_settings': {'care_facility': 0.7}
            }

    async def _assess_paper_relevance(self, paper: Dict, search_concepts: Dict[str, float]) -> float:
        """Assess paper relevance using concepts from AI analysis"""
        try:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            
            score = 0.0
            total_weight = 0.0
            
            # Score based on presence of weighted concepts from each category
            for category in search_concepts.values():
                for concept, weight in category.items():
                    total_weight += weight
                    concept = concept.lower().replace('_', ' ')
                    if concept in title:
                        score += weight * 1.5  # Higher weight for title matches
                    if concept in abstract:
                        score += weight
            
            # Avoid division by zero
            if total_weight == 0:
                return 0.0
            
            # Normalize score
            normalized_score = score / (total_weight * 2.5)  # Maximum possible score
            return min(normalized_score * 10, 10.0)  # Convert to 0-10 scale
            
        except Exception as e:
            logger.error(f"Error assessing paper relevance: {str(e)}")
            return 0.0

    async def _extract_key_points(self, paper: Dict) -> str:
        """Extract key ethical points from paper"""
        try:
            prompt = f"""Extract the key ethical points from this paper:

Title: {paper['title']}
Abstract: {paper['abstract']}

Focus on:
1. Ethical principles discussed
2. Relevant findings
3. Practical recommendations
4. Applicable insights

Return 3-5 key points in bullet form."""

            response = await self._get_gpt_response(prompt)
            return response.strip()
        except:
            return "No key points extracted"

    async def get_relevant_papers(self, query: str, ai_analysis: Dict) -> Dict:
        """Get relevant papers based on query and AI analysis"""
        try:
            # Create prompt for extracting search terms
            search_terms_prompt = """Extract key medical and ethical terms from this case and analysis. Format the response as a strict JSON object with these categories:
            {
                "medical_conditions": [],
                "treatments": [],
                "ethical_principles": [],
                "care_settings": [],
                "key_issues": []
            }

            Case:
            %s

            Analysis:
            %s

            Return ONLY the JSON object, no other text.""" % (query, ai_analysis.get('summary', ''))

            # Get search terms from GPT
            response = await self._get_gpt_response(search_terms_prompt)
            
            try:
                # Clean the response to ensure valid JSON
                cleaned_response = response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                search_terms = json.loads(cleaned_response)
                
                # Log search terms in a readable format
                logger.info("\nExtracted search terms:")
                logger.info("="*50)
                for category, terms in search_terms.items():
                    logger.info(f"\n{category.replace('_', ' ').title()}:")
                    for term in terms:
                        logger.info(f"- {term}")
                logger.info("="*50 + "\n")

                # Search papers using enhanced search
                papers = await self.ethics_db.search_papers(
                    query=query,
                    ai_analysis=ai_analysis,
                    search_terms=search_terms,
                    limit=5
                )
                
                # Add search terms to the response
                return {
                    'papers': papers,
                    'search_terms': search_terms
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing search terms JSON: {str(e)}\nResponse was: {response}")
                return {'papers': [], 'search_terms': {}}
                
        except Exception as e:
            logger.error(f"Error getting relevant papers: {str(e)}")
            return {'papers': [], 'search_terms': {}}

    async def _get_initial_analysis(self, question: str) -> Dict[str, Any]:
        """Get initial AI analysis of the ethical question"""
        try:
            logger.info("\nStarting initial analysis...")
            
            ai_prompt = f"""Analyze this medical ethics question and provide a structured response with EXACTLY these sections:

Question: {question}

SUMMARY
Provide a concise summary of the ethical situation and key principles involved.

RECOMMENDATIONS
- Provide specific, actionable recommendation based on ethical principles
- Include at least 3 clear recommendations
- Support each with ethical reasoning

ETHICAL CONCERNS
- List specific ethical concerns raised by the situation
- Include at least 3 major concerns
- Explain the ethical principles at stake

MITIGATION STRATEGIES
- Provide practical strategies to address the concerns
- Include at least 3 specific strategies
- Explain how each strategy helps

FOLLOW-UP QUESTIONS
- List key questions to better understand the situation
- Include at least 3 important questions
- Focus on gathering relevant ethical information

Use EXACTLY these section headers and bullet points for lists.
"""

            logger.info("\nSending prompt to GPT...")
            ai_response = await self._get_gpt_response(ai_prompt)
            logger.info("\nRaw GPT response:")
            logger.info(f"{ai_response}")
            
            parsed = self.response_parser.parse_gpt_response(ai_response)
            logger.info("\nParsed response:")
            logger.info(f"{json.dumps(parsed, indent=2)}")
            
            return parsed

        except Exception as e:
            logger.error("\nError in initial analysis:")
            logger.error(f"{str(e)}", exc_info=True)
            return {
                'summary': 'Error performing initial analysis',
                'recommendations': [],
                'concerns': [],
                'mitigation_strategies': [],
                'follow_up_questions': []
            } 