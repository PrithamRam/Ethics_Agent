from typing import List, Dict, Optional, Tuple
import sqlite3
import json
import logging
from pathlib import Path
import re
import os
from dotenv import load_dotenv
from openai import OpenAI
import aiosqlite
import asyncio

logger = logging.getLogger(__name__)

class EthicsDatabase:
    def __init__(self):
        self.db_path = "ethics.db"
        self.conn = None
        
        # Initialize OpenAI client
        load_dotenv()
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=openai_key)

    async def get_connection(self):
        """Get or create a database connection"""
        try:
            if not self.conn:
                self.conn = await aiosqlite.connect(self.db_path)
                # Set row factory to return dictionary-like objects
                self.conn.row_factory = aiosqlite.Row
                await self._setup_database()
                # Initialize with some papers if empty
                await self._initialize_with_papers()
            return self.conn
        except Exception as e:
            logger.error(f"Error getting database connection: {str(e)}")
            raise

    async def _setup_database(self):
        """Set up database tables"""
        async with self.conn.cursor() as cursor:
            # Create papers table with all needed columns
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    pubmed_id TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    journal TEXT,
                    year INTEGER,
                    relevance_score REAL
                )
            """)
            
            # Create or recreate FTS table
            await cursor.execute("DROP TABLE IF EXISTS papers_fts")
            await cursor.execute("""
                CREATE VIRTUAL TABLE papers_fts USING fts5(
                    title,
                    abstract,
                    journal,
                    content='papers',
                    content_rowid='rowid',
                    tokenize='porter unicode61'
                )
            """)
            
            await self.conn.commit()
            logger.info("Database tables set up successfully")

    async def _initialize_with_papers(self):
        """Initialize database with papers"""
        try:
            conn = await self.get_connection()
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    pubmed_id TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    journal TEXT,
                    year INTEGER,
                    relevance_score REAL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ethical_considerations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT,
                    consideration TEXT,
                    FOREIGN KEY (paper_id) REFERENCES papers (pubmed_id)
                )
            """)

            # Add indexes for better search performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON papers(title)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_abstract ON papers(abstract)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_considerations ON ethical_considerations(consideration)")

            conn.commit()
            logger.info("Database tables set up successfully")
            
            return True
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            return False

    async def _extract_search_terms(self, query: str) -> List[str]:
        """Extract key ethical and medical concepts from the query using GPT"""
        try:
            prompt = f"""Analyze this medical ethics case and extract key search terms.
            Focus on ethical principles, medical conditions, and relationships.
            Return ONLY a JSON array of search terms, nothing else.

Case:
{query}

Example response format:
["patient confidentiality", "HIV disclosure", "partner notification"]"""

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a medical ethics expert. Return ONLY a JSON array of search terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Clean the response content
            content = response.choices[0].message.content.strip()
            if not content.startswith('[') or not content.endswith(']'):
                logger.warning(f"Unexpected GPT response format: {content}")
                return self._basic_term_extraction(query)
            
            terms = json.loads(content)
            logger.info(f"Successfully extracted {len(terms)} search terms")
            return terms
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return self._basic_term_extraction(query)
        except Exception as e:
            logger.error(f"Error extracting search terms: {str(e)}")
            return self._basic_term_extraction(query)

    def _basic_term_extraction(self, query: str) -> List[str]:
        """Basic term extraction as fallback"""
        # Extract key medical and ethical terms
        key_terms = set()
        
        # Common medical ethics terms
        important_terms = {
            "ethics", "ethical", "consent", "confidential", "privacy",
            "autonomy", "beneficence", "justice", "rights", "dignity",
            "liability", "legal", "obligation", "duty", "care",
            "treatment", "medical", "clinical", "patient", "doctor",
            "hospital", "health", "risk", "harm", "benefit"
        }
        
        # Clean and split query
        words = re.findall(r'\w+', query.lower())
        
        # Find important single words
        key_terms.update(word for word in words if word in important_terms)
        
        # Find important phrases
        text = query.lower()
        phrases = [
            "informed consent", "patient autonomy", "medical ethics",
            "end of life", "quality of life", "standard of care",
            "best interest", "medical decision", "clinical judgment"
        ]
        key_terms.update(phrase for phrase in phrases if phrase in text)
        
        return list(key_terms) if key_terms else ["medical ethics"]

    async def search_papers(self, query: str, limit: int = 10) -> List[Dict]:
        """Search papers using extracted terms"""
        try:
            # Extract search terms
            search_terms = await self._extract_search_terms(query)
            if not search_terms:
                logger.warning("No search terms extracted")
                return []
            
            # Build search query
            fts_query = self._build_simple_query(search_terms)
            logger.info(f"Searching with query: {fts_query}")
            
            # Get database connection
            conn = await self.get_connection()
            
            # Execute search with broader matching
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    WITH matching_papers AS (
                        SELECT papers.rowid, papers.pubmed_id, papers.title, papers.abstract, 
                               papers.journal, papers.year
                        FROM papers
                        JOIN papers_fts ON papers.rowid = papers_fts.rowid
                        WHERE papers_fts MATCH ?
                    )
                    SELECT p.pubmed_id, p.title, p.abstract, p.journal, p.year,
                        CASE 
                            WHEN lower(p.title) LIKE '%ethic%' OR
                                 lower(p.title) LIKE '%dementia%' OR
                                 lower(p.title) LIKE '%care%' THEN 3
                            WHEN lower(p.journal) LIKE '%ethic%' THEN 2
                            WHEN lower(p.abstract) LIKE '%ethic%' OR
                                 lower(p.abstract) LIKE '%dementia%' OR
                                 lower(p.abstract) LIKE '%care%' THEN 1
                            ELSE 0
                        END as relevance
                    FROM matching_papers p
                    UNION
                    SELECT p.pubmed_id, p.title, p.abstract, p.journal, p.year,
                        CASE 
                            WHEN lower(p.title) LIKE '%ethic%' OR
                                 lower(p.title) LIKE '%dementia%' OR
                                 lower(p.title) LIKE '%care%' THEN 3
                            WHEN lower(p.journal) LIKE '%ethic%' THEN 2
                            WHEN lower(p.abstract) LIKE '%ethic%' OR
                                 lower(p.abstract) LIKE '%dementia%' OR
                                 lower(p.abstract) LIKE '%care%' THEN 1
                            ELSE 0
                        END as relevance
                    FROM papers p
                    WHERE lower(p.title) LIKE '%dementia%'
                       OR lower(p.title) LIKE '%care%'
                       OR lower(p.abstract) LIKE '%dementia%'
                       OR lower(p.abstract) LIKE '%care%'
                    ORDER BY relevance DESC, year DESC
                    LIMIT ?
                """, (fts_query, limit))
                
                rows = await cursor.fetchall()
                results = []
                for row in rows:
                    results.append({
                        'title': row['title'],
                        'abstract': row['abstract'],
                        'journal': row['journal'],
                        'year': row['year'],
                        'relevance': row['relevance'],
                        'pubmed_id': row['pubmed_id']
                    })
                
                logger.info(f"\nFound {len(results)} matching papers")
                if results:
                    logger.info("\nFirst matching paper:")
                    logger.info(f"Title: {results[0]['title']}")
                    logger.info(f"Journal: {results[0]['journal']}")
                    logger.info(f"Year: {results[0]['year']}")
                    logger.info(f"Relevance: {results[0]['relevance']}")
                
                return results
            
        except Exception as e:
            logger.error(f"Error searching papers: {str(e)}")
            return []

    def _build_simple_query(self, terms: List[str]) -> str:
        """Build simple FTS query"""
        query_parts = []
        for term in terms:
            clean_term = term.strip().lower()
            clean_term = re.sub(r'[^\w\s]', ' ', clean_term).strip()
            
            if not clean_term:
                continue
            
            if ' ' in clean_term:
                query_parts.append(f'"{clean_term}"')  # Exact phrase
            else:
                query_parts.append(clean_term)  # Single word
        
        return ' OR '.join(query_parts)

    def close(self):
        """Close the database connection"""
        self.conn.close()

    def rebuild_fts_index(self):
        """Rebuild the FTS index"""
        try:
            cursor = self.conn.cursor()
            logger.info("Starting FTS index rebuild...")
            
            # Insert all papers into FTS
            cursor.execute("""
                INSERT INTO papers_fts(rowid, title, abstract, journal)
                SELECT rowid,
                       title,
                       abstract,
                       journal
                FROM papers
            """)
            
            self.conn.commit()
            logger.info("FTS index rebuild completed successfully")
            
        except Exception as e:
            logger.error(f"Error rebuilding FTS index: {str(e)}")
            raise

    @classmethod
    async def initialize_database_if_needed(cls):
        """Initialize the database if it doesn't exist"""
        db = cls()
        await db.get_connection()
        return db

    def _build_column_query(self, terms: List[str], column: str) -> str:
        """Build column-specific FTS query"""
        query_parts = []
        for term in terms:
            clean_term = term.strip().lower()
            clean_term = re.sub(r'[^\w\s]', ' ', clean_term).strip()
            
            if not clean_term:
                continue
            
            if ' ' in clean_term:
                query_parts.append(f'{column}:"{clean_term}"')
            else:
                query_parts.append(f'{column}:{clean_term}')
        
        return ' OR '.join(query_parts) 