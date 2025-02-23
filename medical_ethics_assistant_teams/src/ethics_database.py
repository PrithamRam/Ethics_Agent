from typing import List, Dict
import sqlite3
import json
from dataclasses import asdict
from src.document_processor import Reference
import aiosqlite
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EthicsDatabase:
    def __init__(self):
        # Create data directory if it doesn't exist
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        self.db_path = str(data_dir / 'ethics_references.db')
        self._connection = None
    
    async def get_connection(self):
        """Get or create database connection"""
        if self._connection is None:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(exist_ok=True)
            
            self._connection = await aiosqlite.connect(self.db_path)
            await self.create_tables(self._connection)
            # Initialize with default data if empty
            await self._initialize_if_empty()
        return self._connection
    
    async def close(self):
        """Close the database connection"""
        if self._connection:
            await self._connection.close()
            self._connection = None
    
    async def create_tables(self, db):
        """Create necessary database tables"""
        await db.execute('''
            CREATE TABLE IF NOT EXISTS ref_entries (
                id INTEGER PRIMARY KEY,
                pubmed_id TEXT UNIQUE,
                title TEXT,
                abstract TEXT,
                keywords TEXT,
                ethical_considerations TEXT
            )
        ''')
        
        await db.execute('''
            CREATE TABLE IF NOT EXISTS ethical_principles (
                id INTEGER PRIMARY KEY,
                principle TEXT UNIQUE,
                description TEXT,
                reference_ids TEXT
            )
        ''')
        
        await db.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS ref_entries_fts USING fts5(
                pubmed_id,
                title,
                abstract,
                keywords,
                ethical_considerations,
                content='ref_entries'
            )
        ''')
    
    async def populate_database(self, data: Dict):
        """Populate database with processed data"""
        conn = await self.get_connection()
        async with conn.cursor() as cursor:
            for ref in data["references"]:
                await self._insert_reference(cursor, ref)
            for principle, details in data["principles"].items():
                await self._insert_principle(cursor, principle, details)
            await conn.commit()
    
    async def _insert_reference(self, cursor, reference: Reference):
        """Insert a reference into the database"""
        ref_dict = asdict(reference)
        ref_dict['keywords'] = json.dumps(ref_dict['keywords'])
        ref_dict['ethical_considerations'] = json.dumps(ref_dict['ethical_considerations'])
        
        await cursor.execute('''
            INSERT OR REPLACE INTO ref_entries 
            (pubmed_id, title, abstract, keywords, ethical_considerations)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            ref_dict['pubmed_id'],
            ref_dict['title'],
            ref_dict['abstract'],
            ref_dict['keywords'],
            ref_dict['ethical_considerations']
        ))
        
        # Update FTS table
        await cursor.execute('''
            INSERT OR REPLACE INTO ref_entries_fts 
            (pubmed_id, title, abstract, keywords, ethical_considerations)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            ref_dict['pubmed_id'],
            ref_dict['title'],
            ref_dict['abstract'],
            ref_dict['keywords'],
            ref_dict['ethical_considerations']
        ))
    
    async def _insert_principle(self, cursor, principle: str, details: Dict):
        """Insert an ethical principle into the database"""
        await cursor.execute('''
            INSERT OR REPLACE INTO ethical_principles 
            (principle, description, reference_ids)
            VALUES (?, ?, ?)
        ''', (
            principle,
            details['description'],
            json.dumps([ref['pubmed_id'] for ref in details['contexts']])
        ))
    
    async def list_all_papers(self) -> List[Dict]:
        """List all papers in the database"""
        conn = await self.get_connection()
        try:
            async with conn.cursor() as cursor:
                await cursor.execute('SELECT * FROM ref_entries')
                rows = await cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error listing papers: {str(e)}")
            return []

    async def add_paper(self, pubmed_id: str, title: str, abstract: str, 
                       keywords: List[str], ethical_considerations: List[str]):
        """Add a paper to the database"""
        conn = await self.get_connection()
        try:
            async with conn.cursor() as cursor:
                await cursor.execute('''
                    INSERT OR REPLACE INTO ref_entries 
                    (pubmed_id, title, abstract, keywords, ethical_considerations)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    pubmed_id,
                    title,
                    abstract,
                    json.dumps(keywords),
                    json.dumps(ethical_considerations)
                ))
                await conn.commit()
                logger.info(f"Added paper {pubmed_id} to database")
                return True
        except Exception as e:
            logger.error(f"Error adding paper: {str(e)}")
            return False

    async def search_relevant_references(self, query: str) -> List[Dict]:
        """Search for relevant references in local database"""
        conn = await self.get_connection()
        try:
            async with conn.cursor() as cursor:
                # Log current database contents
                await cursor.execute('SELECT COUNT(*) FROM ref_entries')
                count = await cursor.fetchone()
                logger.info(f"Total papers in database: {count[0]}")

                # Create search terms from medical/ethical concepts
                search_terms = query.lower().split()
                medical_terms = ['ventilator', 'covid', 'triage', 'resource', 'allocation', 'ethics', 'ethical']
                search_terms = [term for term in search_terms if any(med_term in term for med_term in medical_terms)]
                
                if not search_terms:
                    # If no medical terms found, use the whole query
                    search_terms = [query.lower()]

                # Construct the search conditions
                search_conditions = []
                params = []
                
                for term in search_terms:
                    search_conditions.extend([
                        "LOWER(title) LIKE ?",
                        "LOWER(abstract) LIKE ?",
                        "LOWER(keywords) LIKE ?",
                        "LOWER(ethical_considerations) LIKE ?"
                    ])
                    term_pattern = f"%{term}%"
                    params.extend([term_pattern] * 4)

                # Construct the query
                sql = f'''
                    SELECT DISTINCT r.* 
                    FROM ref_entries r 
                    WHERE {" OR ".join(search_conditions)}
                '''
                
                logger.info(f"Executing search with terms: {search_terms}")
                await cursor.execute(sql, params)
                
                rows = await cursor.fetchall()
                logger.info(f"Found {len(rows)} matching papers")
                
                results = []
                for row in rows:
                    ref = self._row_to_dict(row)
                    ref['relevance_score'] = self._calculate_relevance(
                        query, 
                        ref['title'] + ' ' + ' '.join(ref['ethical_considerations'])
                    )
                    results.append(ref)
                
                # Sort by relevance
                results.sort(key=lambda x: x['relevance_score'], reverse=True)
                return results
                
        except Exception as e:
            logger.error(f"Error searching references: {str(e)}")
            return []
    
    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score based on term matching"""
        query_terms = set(self._prepare_search_terms(query))
        text_terms = set(self._prepare_search_terms(text))
        
        matches = query_terms.intersection(text_terms)
        return len(matches) / len(query_terms) if query_terms else 0
    
    def _prepare_search_terms(self, query: str) -> List[str]:
        """Prepare search terms from query"""
        # Split query into words and remove common words
        words = re.findall(r'\w+', query.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
        terms = [word for word in words if word not in stop_words]
        return terms
    
    def _row_to_dict(self, row) -> Dict:
        """Convert a database row to a dictionary"""
        return {
            'pubmed_id': row[1],
            'title': row[2],
            'abstract': row[3],
            'keywords': json.loads(row[4]),
            'ethical_considerations': json.loads(row[5]),
            'pubmed_url': f"https://pubmed.ncbi.nlm.nih.gov/{row[1]}/",
            'citation': f"PMID: {row[1]}",
        }
    
    async def get_sample_question(self) -> str:
        """Get a sample question based on database content"""
        try:
            conn = await self.get_connection()
            async with conn.cursor() as cursor:
                # Query for a random ethical consideration
                await cursor.execute('''
                    SELECT r.title, r.ethical_considerations 
                    FROM ref_entries r 
                    WHERE r.ethical_considerations IS NOT NULL 
                    ORDER BY RANDOM() 
                    LIMIT 1
                ''')
                
                result = await cursor.fetchone()
                if result:
                    title, considerations = result
                    # Generate a question based on the content
                    return f"What are the ethical implications of {title.lower()}?"
                
                return "What are the ethical considerations for resource allocation during health crises?"
        except Exception as e:
            logger.error(f"Error getting sample question: {str(e)}")
            return "What are the key ethical principles in medical decision-making?"

    async def initialize_with_paper(self):
        """Initialize database with ventilator ethics papers"""
        papers = [
            {
                'pubmed_id': '32381261',
                'title': 'Ethics and law in the intensive care unit',
                'abstract': 'The COVID-19 pandemic has created an unprecedented challenge for intensive care...',
                'keywords': ['COVID-19', 'ventilators', 'ethics', 'resource allocation', 'triage'],
                'ethical_considerations': [
                    'Fair allocation of scarce resources',
                    'Legal framework for rationing decisions',
                    'Ethical principles in triage',
                    'Protection of healthcare workers',
                    'Documentation requirements'
                ]
            },
            {
                'pubmed_id': '32289415',
                'title': 'Fair Allocation of Scarce Medical Resources in the Time of Covid-19',
                'abstract': 'The Covid-19 pandemic has created an unprecedented challenge for allocating scarce medical resources...',
                'keywords': ['resource allocation', 'medical ethics', 'COVID-19', 'ventilators'],
                'ethical_considerations': [
                    'Maximizing benefits',
                    'Treating people equally',
                    'Promoting instrumental value',
                    'Giving priority to the worst off'
                ]
            }
        ]
        
        conn = await self.get_connection()
        try:
            async with conn.cursor() as cursor:
                for paper in papers:
                    await self.add_paper(
                        paper['pubmed_id'],
                        paper['title'],
                        paper['abstract'],
                        paper['keywords'],
                        paper['ethical_considerations']
                    )
            logger.info(f"Initialized database with {len(papers)} papers")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")

    async def _initialize_if_empty(self):
        """Initialize database with default data if empty"""
        async with self._connection.cursor() as cursor:
            await cursor.execute("SELECT COUNT(*) FROM ref_entries")
            count = await cursor.fetchone()
            if count[0] == 0:
                await self.initialize_with_paper()
                await self._connection.commit() 