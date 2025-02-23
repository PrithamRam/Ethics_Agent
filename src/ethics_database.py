from typing import List, Dict, Optional, Tuple
import sqlite3
import json
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class EthicsDatabase:
    # Add stop_words as a class variable
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
        'to', 'was', 'were', 'will', 'with', 'what', 'when', 'where', 
        'who', 'why', 'how', 'after', 'before', 'during', 'these', 'those',
        'this', 'there', 'here', 'have', 'had', 'been', 'would', 'should',
        'could', 'may', 'might', 'must', 'shall', 'we', 'our', 'ours', 'us',
        'they', 'them', 'their', 'theirs', 'do', 'does', 'did', 'doing',
        'get', 'gets', 'got', 'than', 'then', 'need', 'needs', 'into'
    }

    def __init__(self, db_path: str = "ethics.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def setup_database(self):
        """Create the necessary tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Drop existing tables if they exist
        cursor.execute("DROP TABLE IF EXISTS ref_entries")
        cursor.execute("DROP TABLE IF EXISTS ref_entries_fts")
        cursor.execute("DROP TABLE IF EXISTS ethical_principles")
        cursor.execute("DROP TABLE IF EXISTS keywords")
        cursor.execute("DROP TABLE IF EXISTS ethical_considerations")
        cursor.execute("DROP TABLE IF EXISTS papers")
        
        # Create papers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                pubmed_id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                year INTEGER,
                journal TEXT
            )
        """)
        
        # Create keywords table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                paper_id TEXT,
                keyword TEXT,
                FOREIGN KEY(paper_id) REFERENCES papers(pubmed_id)
            )
        """)
        
        # Create ethical_considerations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ethical_considerations (
                paper_id TEXT,
                consideration TEXT,
                FOREIGN KEY(paper_id) REFERENCES papers(pubmed_id)
            )
        """)
        
        # Add FTS5 virtual table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
                pubmed_id UNINDEXED,
                title,
                abstract,
                year UNINDEXED,
                journal,
                content='papers',
                content_rowid='rowid'
            )
        """)
        
        self.conn.commit()

    def add_paper(self, **kwargs):
        """Add a paper and its associated data to the database"""
        cursor = self.conn.cursor()
        
        try:
            # Extract fields from kwargs
            pubmed_id = kwargs.get('pubmed_id')
            title = kwargs.get('title')
            abstract = kwargs.get('abstract')
            year = kwargs.get('year')
            journal = kwargs.get('journal')
            keywords = kwargs.get('keywords', [])
            ethical_considerations = kwargs.get('ethical_considerations', [])
            
            # Insert paper into main table
            cursor.execute("""
                INSERT OR REPLACE INTO papers (pubmed_id, title, abstract, year, journal)
                VALUES (?, ?, ?, ?, ?)
            """, (pubmed_id, title, abstract, year, journal))
            
            # Insert into FTS table
            cursor.execute("""
                INSERT OR REPLACE INTO papers_fts (pubmed_id, title, abstract, year, journal)
                VALUES (?, ?, ?, ?, ?)
            """, (pubmed_id, title, abstract, year, journal))
            
            # Clear existing keywords and considerations
            cursor.execute("DELETE FROM keywords WHERE paper_id = ?", (pubmed_id,))
            cursor.execute("DELETE FROM ethical_considerations WHERE paper_id = ?", (pubmed_id,))
            
            # Insert keywords if any
            if keywords:
                cursor.executemany("""
                    INSERT INTO keywords (paper_id, keyword)
                    VALUES (?, ?)
                """, [(pubmed_id, kw) for kw in keywords])
            
            # Insert ethical considerations if any
            if ethical_considerations:
                cursor.executemany("""
                    INSERT INTO ethical_considerations (paper_id, consideration)
                    VALUES (?, ?)
                """, [(pubmed_id, ec) for ec in ethical_considerations])
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error adding paper {kwargs.get('pubmed_id')}: {str(e)}")
            self.conn.rollback()

    def search_papers(self, query: str, limit: int = 10) -> List[Dict]:
        """Enhanced search with FTS5 and intelligent term extraction"""
        try:
            cursor = self.conn.cursor()
            
            # Extract key terms using GPT
            search_terms = self._extract_search_terms(query)
            
            # Build FTS query with extracted terms
            fts_query = ' OR '.join(f'"{term}"*{weight}' for term, weight in search_terms)
            
            sql = """
                WITH scored_papers AS (
                    SELECT 
                        p.*,
                        rank * -1 as relevance_score,  -- Convert rank to positive score
                        GROUP_CONCAT(DISTINCT k.keyword) as keywords,
                        GROUP_CONCAT(DISTINCT e.consideration) as ethical_considerations
                    FROM papers_fts p
                    LEFT JOIN keywords k ON p.pubmed_id = k.paper_id
                    LEFT JOIN ethical_considerations e ON p.pubmed_id = e.paper_id
                    WHERE papers_fts MATCH ?
                    GROUP BY p.pubmed_id
                )
                SELECT *,
                    CASE 
                        WHEN title LIKE ? THEN relevance_score * 2
                        WHEN abstract LIKE ? THEN relevance_score * 1.5
                        ELSE relevance_score
                    END as final_score
                FROM scored_papers
                ORDER BY final_score DESC, year DESC
                LIMIT ?
            """
            
            params = [
                fts_query,
                f"%{query}%",
                f"%{query}%",
                limit
            ]
            
            logger.info(f"Searching with FTS query: {fts_query}")
            cursor.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                paper = dict(row)
                paper['keywords'] = paper['keywords'].split(',') if paper['keywords'] else []
                paper['ethical_considerations'] = paper['ethical_considerations'].split(',') if paper['ethical_considerations'] else []
                paper['relevance_score'] = paper['final_score']
                results.append(paper)
                logger.info(f"Found paper: {paper['title']} (score: {paper['relevance_score']})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching papers: {str(e)}")
            return []

    def _extract_search_terms(self, query: str) -> List[Tuple[str, int]]:
        """Use GPT to extract and weight search terms"""
        try:
            prompt = f"""Analyze this medical ethics query and extract key search terms:
            "{query}"
            
            1. Identify main concepts and their synonyms
            2. Include both specific terms and broader ethical concepts
            3. Consider medical terminology and ethical principles
            4. Return terms in order of relevance (most important first)
            
            Format each term as: term (weight 1-3)
            Example: ethics (3), consent (2), treatment (1)"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a medical ethics expert helping to identify key search terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Parse response into terms and weights
            terms_text = response.choices[0].message.content
            terms = []
            for match in re.finditer(r'(\w+)\s*\((\d)\)', terms_text):
                term, weight = match.groups()
                terms.append((term.lower(), int(weight)))
            
            # Add any exact phrases from query that contain key terms
            phrases = re.findall(r'"([^"]+)"', query)
            for phrase in phrases:
                if any(term[0] in phrase.lower() for term in terms):
                    terms.append((phrase.lower(), 3))
            
            return terms
            
        except Exception as e:
            logger.error(f"Error extracting search terms: {str(e)}")
            # Fallback to basic term extraction
            words = query.lower().split()
            return [(w, 1) for w in words if w not in self.stop_words and len(w) > 3]

    def close(self):
        """Close the database connection"""
        self.conn.close()

    def rebuild_fts_index(self):
        """Rebuild the FTS index from papers table"""
        cursor = self.conn.cursor()
        try:
            # Clear FTS table
            cursor.execute("DELETE FROM papers_fts")
            
            # Copy data from papers table
            cursor.execute("""
                INSERT INTO papers_fts (pubmed_id, title, abstract, year, journal)
                SELECT pubmed_id, title, abstract, year, journal FROM papers
            """)
            
            self.conn.commit()
            logger.info("FTS index rebuilt successfully")
        except Exception as e:
            logger.error(f"Error rebuilding FTS index: {str(e)}")
            self.conn.rollback()

    @classmethod
    def initialize_database_if_needed(cls):
        """Initialize database only if it doesn't exist"""
        db = cls()
        db_file = Path("ethics.db")
        
        if db_file.exists():
            try:
                cursor = db.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM papers")
                count = cursor.fetchone()[0]
                logger.info(f"Using existing database with {count} papers")
                
                # Rebuild FTS index
                db.rebuild_fts_index()
                
                return db
            except sqlite3.OperationalError as e:
                logger.info(f"Database error ({str(e)}), recreating...")
                db.conn.close()
                db_file.unlink()
        
        # Create new database
        db = cls()  # Create fresh connection
        logger.info("Creating new database...")
        
        # Create tables
        db.setup_database()
        
        # Load papers from JSON
        data_file = Path('data') / 'pubmed_ethics_papers.json'
        if not data_file.exists():
            logger.error(f"Papers file not found at {data_file.absolute()}")
            return None
            
        logger.info(f"Loading papers from {data_file}")
        with open(data_file) as f:
            papers = json.load(f)
        logger.info(f"Loaded {len(papers)} papers")
        
        # Populate database
        logger.info("Adding papers to database...")
        for paper in papers:
            try:
                db.add_paper(**paper)  # Use kwargs expansion
            except Exception as e:
                logger.error(f"Error adding paper {paper.get('pubmed_id')}: {str(e)}")
                continue
        
        logger.info("Database initialization complete")
        return db 