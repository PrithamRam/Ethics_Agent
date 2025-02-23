from typing import List, Dict
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
    
    async def get_ethical_guidance(self, query: str) -> Dict:
        """Get ethical guidance for a medical ethics query"""
        try:
            # Store the initial query and context
            self.current_query_context = {
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
            
            # First search local database
            local_refs = await self.ethics_db.search_relevant_references(query)
            logger.info(f"Found {len(local_refs)} relevant papers in local database")

            # Then search PubMed for additional references
            pmids, total_results = await self.pubmed_handler.search_pubmed(query)
            pubmed_refs = await self.pubmed_handler.fetch_papers(pmids)
            
            # Combine and deduplicate references
            all_refs = self._merge_references(local_refs, pubmed_refs)
            
            # Add to conversation history
            self.conversation_history.append({
                'type': 'initial_query',
                'query': query,
                'references': all_refs,
                'timestamp': self.current_query_context['timestamp']
            })
            
            # Get GPT response with context from references
            context = self._prepare_context(all_refs, [])
            guidance_response = await self._get_gpt_response(
                f"""Based on these references and ethical guidelines, analyze the following query:
                
                Query: {query}
                
                Context:
                {context}
                
                Please provide a structured response with:
                1. Summary
                2. Key Recommendations
                3. Ethical Concerns
                4. Mitigation Strategies
                5. Follow-up Questions
                """
            )
            
            # Parse the response
            analysis = self.response_parser.parse_gpt_response(guidance_response)
            
            # Generate HTML response
            html = self.template_manager.generate_response_html(
                query=query,
                analysis=analysis,
                references=all_refs,
                total_results=len(local_refs) + (total_results or 0)
            )
            
            return {
                'html': html,
                'analysis': analysis,
                'references': all_refs,
                'total_results': len(local_refs) + (total_results or 0),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error generating ethical guidance: {str(e)}")
            return {
                'html': self.template_manager.generate_error_html(str(e)),
                'error': str(e),
                'status': 'failed'
            }
    
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
            context_parts.append(f"Ethical Considerations:")
            for consideration in ref['ethical_considerations']:
                context_parts.append(f"- {consideration}")
            context_parts.append("")
        
        # Add PubMed context
        for paper in pubmed_context:
            context_parts.append(f"PubMed Paper {paper['pubmed_id']}:")
            context_parts.append(f"Title: {paper['title']}")
            context_parts.append(f"Abstract: {paper['abstract']}")
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
            # Prepare messages for the chat
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": str(self.current_query_context)},
                {"role": "assistant", "content": str(self.last_response) if self.last_response else ""},
                {"role": "user", "content": f"""Please provide a structured response to this follow-up question:
                    {question}
                    
                    Format your response with:
                    1. Summary of the answer
                    2. Key recommendations or points
                    3. Any relevant follow-up questions"""}
            ]

            # Get AI response using synchronous call
            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            # Extract response content
            response_content = completion.choices[0].message.content

            # Parse the response
            analysis = self.response_parser.parse_gpt_response(response_content)
            
            # Get relevant references
            references = await self.pubmed_handler.get_relevant_papers(question)

            # Store this as the last response
            self.last_response = response_content

            return {
                'analysis': analysis,
                'references': references,
                'status': 'success',
                'response': response_content
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