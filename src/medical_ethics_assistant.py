from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from openai import OpenAI
from src.pubmed_handler import PubMedHandler
from src.ethics_database import EthicsDatabase
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
        self.client = OpenAI(api_key=self.openai_key)  # Create client instance
        
        self.pubmed_handler = PubMedHandler(self.config)
        self.ethics_db = EthicsDatabase()
        self.doc_processor = DocumentProcessor()
        self.response_parser = ResponseParser()
        self.template_manager = TemplateManager()
        self.conversation_history = []  # Store conversation history
        self.current_query_context = None  # Store the current query context
        self.last_response = None  # Store the last response
    
    @classmethod
    async def create(cls, config: SystemConfig = None):
        """Factory method to create and initialize the assistant"""
        assistant = cls(config)
        # Get connection will initialize the database
        await assistant.ethics_db.get_connection()
        return assistant
    
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
            
            # Add system message
            messages.append({
                "role": "system",
                "content": "You are an AI assistant specializing in medical ethics. Provide detailed, structured analysis of ethical considerations."
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
                max_tokens=1000
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
    
    async def get_ethical_guidance(self, question: str) -> Dict[str, Any]:
        """Get ethical guidance for a medical ethics question"""
        try:
            # Get relevant papers
            logger.info(f"Searching papers for question: {question}")
            try:
                papers = await self.ethics_db.search_relevant_references(question)
            except Exception as db_error:
                logger.error(f"Database error: {str(db_error)}")
                raise ValueError(f"Failed to search database: {str(db_error)}")

            if not papers and isinstance(papers, list):
                logger.info("No papers found in database")
            
            # Structure for both terminal logging and HTML response
            literature_analysis = {
                'papers': papers if papers else []
            }

            # Log literature analysis in same format as HTML will display
            logger.info("\n=== Literature Analysis ===")
            for paper in papers:
                logger.info(f"\nPaper: {paper['title']}\n")
                logger.info(f"Abstract: {paper['abstract']}\n")
                logger.info(f"Ethical Considerations: {paper['ethical_considerations']}\n")
                logger.info(f"Keywords: {paper['keywords']}\n")
                logger.info(f"Relevance Score: {paper['relevance_score']}\n")

            # Get AI analysis with literature context using the SAME papers
            literature_context = "\n".join([
                f"""
                Title: {paper['title']}
                Ethical Considerations: {', '.join(paper['ethical_considerations'])}
                Keywords: {', '.join(paper['keywords'])}
                Relevance Score: {paper['relevance_score']}
                """ for paper in papers
            ]) if papers else "No specific literature found."

            ai_prompt = f"""Based on both the literature and general medical ethics principles, analyze the following query:

                    Query: {question}

                    Relevant Literature:
                    {literature_context}

                    Please provide a structured response with:
                    1. Summary (including insights from literature if available)
                    2. Key Recommendations (based on literature and principles)
                    3. Ethical Concerns
                    4. Mitigation Strategies
                    5. Follow-up Questions
                    """
            
            # Get and log raw AI response
            ai_response = await self._get_gpt_response(ai_prompt)
            logger.info("\n=== AI Analysis ===")
            logger.info(f"\nAI Response: {ai_response}\n")

            # Parse AI response into sections
            sections = {
                'summary': '',
                'recommendations': [],
                'concerns': [],
                'strategies': [],
                'questions': []
            }

            current_section = None
            section_text = []
            
            for line in ai_response.split('\n'):
                line = line.strip()
                if not line:
                    continue

                if '1. Summary' in line:
                    current_section = 'summary'
                elif '2. Key Recommendations' in line:
                    sections['summary'] = '\n'.join(section_text).strip()
                    section_text = []
                    current_section = 'recommendations'
                elif '3. Ethical Concerns' in line:
                    current_section = 'concerns'
                elif '4. Mitigation Strategies' in line:
                    current_section = 'strategies'
                elif '5. Follow-up Questions' in line:
                    current_section = 'questions'
                elif line and current_section:
                    if current_section == 'summary':
                        section_text.append(line)
                    elif line.startswith('-') or line.startswith('•'):
                        sections[current_section].append(line[1:].strip())
                    elif not any(marker in line for marker in ['1.', '2.', '3.', '4.', '5.']):
                        if current_section == 'summary':
                            section_text.append(line)
                        else:
                            sections[current_section].append(line.strip())

            if current_section == 'summary':
                sections['summary'] = '\n'.join(section_text).strip()

            # Log structured response in same format as HTML
            logger.info("\n=== Structured AI Response ===\n")
            logger.info("SUMMARY:")
            logger.info(sections['summary'])
            logger.info("\nRECOMMENDATIONS:")
            for r in sections['recommendations']:
                logger.info(f"• {r}")
            logger.info("\nCONCERNS:")
            for c in sections['concerns']:
                logger.info(f"• {c}")
            logger.info("\nSTRATEGIES:")
            for s in sections['strategies']:
                logger.info(f"• {s}")
            logger.info("\nQUESTIONS:")
            for q in sections['questions']:
                logger.info(f"• {q}")

            # Return the exact same data we logged
            response_data = {
                'status': 'success',
                'literature_analysis': literature_analysis,
                'ai_analysis': sections
            }
            
            # Log what we're sending to HTML
            logger.info(f"\nSending to HTML: {json.dumps(response_data, indent=2)}")
            
            return response_data

        except Exception as e:
            error_msg = f"Error in get_ethical_guidance: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'status': 'error',
                'error': error_msg,
                'literature_analysis': {'papers': []},
                'ai_analysis': None
            }
    
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