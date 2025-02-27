from typing import List, Dict
import logging
import sqlite3
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from elasticsearch import AsyncElasticsearch
import ssl
import certifi

logger = logging.getLogger(__name__)

class EthicsDB:
    def __init__(self):
        load_dotenv()
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.openai_key)
        self.db_path = 'ethics.db'  # Path to your SQLite database
        # Only initialize PubMed handler if needed
        self.pubmed_handler = None
        
        # Define ethics keywords
        self.ethics_keywords = [
            # Core ethical principles
            'ethical', 'ethics', 'moral', 'bioethics', 'autonomy',
            'beneficence', 'justice', 'consent', 'privacy', 'dignity',
            'rights', 'fairness', 'equity', 'discrimination',
            
            # Medical ethics specific
            'informed consent', 'confidentiality', 'patient autonomy',
            'medical ethics', 'clinical ethics', 'research ethics',
            'professional ethics', 'healthcare ethics',
            
            # Decision making
            'decision making', 'capacity', 'competency', 'surrogate',
            'advance directive', 'living will', 'power of attorney',
            
            # Care considerations
            'quality of life', 'end of life', 'palliative', 'futility',
            'best interest', 'standard of care', 'duty of care',
            
            # Research ethics
            'institutional review', 'research subject', 'clinical trial',
            'human subject', 'research integrity', 'data protection',
            
            # Social justice
            'access to care', 'health disparity', 'vulnerable population',
            'social justice', 'resource allocation', 'healthcare access'
        ]

        # Database configuration
        self.MIN_ETHICAL_CONSIDERATIONS = 2  # Minimum ethical considerations required per paper

        # Load Elasticsearch config
        config_path = Path('config/elasticsearch_config.txt')
        if config_path.exists():
            with open(config_path) as f:
                config = {}
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            key, value = line.split('=', 1)
                            config[key.strip()] = value.strip()
                        except ValueError:
                            continue  # Skip lines that don't have key=value format
        
            # Initialize Elasticsearch with config values
            self.es = AsyncElasticsearch(
                [config['HOST']],
                basic_auth=(config['USERNAME'], config['PASSWORD']),
                verify_certs=False,  # Disable certificate verification
                ssl_context=None  # Remove SSL context since we're not verifying certs
            )
            self.index_name = config['INDEX_NAME']
        else:
            logger.warning("Elasticsearch config not found, using defaults")
            # Fall back to default values
            self.es = AsyncElasticsearch(
                ['https://localhost:9200'],
                basic_auth=('elastic', 'jskQiFwLICPVn00Zm6mU'),
                verify_certs=False,  # Disable certificate verification
                ssl_context=None  # Remove SSL context
            )
            self.index_name = 'medical_ethics'

    async def _get_gpt_response(self, prompt: str) -> str:
        """Get response from GPT model"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a medical ethics expert. Extract key concepts from queries. Always respond in JSON array format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting GPT response: {str(e)}")
            raise

    async def search_papers(self, query: str, ai_analysis: Dict = None, search_terms: Dict = None, limit: int = 5) -> List[Dict]:
        """Search papers using local SQLite database"""
        try:
            conn = await self.get_connection()
            cursor = conn.cursor()

            # First get total count of papers in database
            cursor.execute("SELECT COUNT(*) FROM papers")
            total_papers = cursor.fetchone()[0]
            
            logger.info(f"Searching {total_papers} papers in local ethics database")

            # For database search, only use medical terms since papers are pre-filtered for ethics
            medical_terms = []
            if search_terms:
                medical_terms.extend(search_terms.get('medical_conditions', []))
                medical_terms.extend(search_terms.get('treatments', []))
            
            if not medical_terms:
                logger.warning("No medical terms found for search")
                return []

            # Create search conditions for each term
            search_conditions = []
            score_components = []
            params = []
            search_words = set()

            logger.info(f"Searching database with medical terms: {medical_terms}")
            
            # First pass - add exact phrase matches with higher scores
            for term in medical_terms:
                term = term.lower()
                # Add parameters for score component with higher weights
                params.extend([f'%{term}%', f'%{term}%'])
                score_components.append(
                    f"CASE WHEN LOWER(p.title) LIKE ? THEN 20 "  # Increased from 10
                    f"WHEN LOWER(p.abstract) LIKE ? THEN 10 ELSE 0 END"  # Increased from 5
                )
                # Add same parameters again for WHERE clause
                params.extend([f'%{term}%', f'%{term}%'])
                search_conditions.append("(LOWER(p.title) LIKE ? OR LOWER(p.abstract) LIKE ?)")
                search_words.add(term)

            # Add ethical terms to boost relevance
            if search_terms and search_terms.get('ethical_principles'):
                for term in search_terms['ethical_principles']:
                    term = term.lower()
                    # Add parameters for both score and WHERE clause
                    params.extend([f'%{term}%', f'%{term}%'])  # For score
                    params.extend([f'%{term}%', f'%{term}%'])  # For WHERE
                    score_components.append(
                        f"CASE WHEN LOWER(p.title) LIKE ? THEN 15 "
                        f"WHEN LOWER(p.abstract) LIKE ? THEN 8 ELSE 0 END"
                    )
                    search_conditions.append("(LOWER(p.title) LIKE ? OR LOWER(p.abstract) LIKE ?)")
                    search_words.add(term)

            # Debug logging
            logger.info(f"Search parameters:")
            logger.info(f"- Terms: {sorted(list(search_words))}")
            logger.info(f"- Score components: {len(score_components)}")
            logger.info(f"- Search conditions: {len(search_conditions)}")
            logger.info(f"- Parameters: {len(params)}")

            # Get papers with any relevance score > 0
            sql = """
            SELECT DISTINCT 
                p.pubmed_id,
                p.title,
                p.authors,
                p.abstract,
                p.journal,
                p.year,
                p.keywords,
                ({}) as relevance_score
            FROM papers p
            WHERE {}
            ORDER BY relevance_score DESC
            LIMIT ?
            """.format(
                " + ".join(score_components), 
                " OR ".join(search_conditions)
            )
            
            params.append(limit)
            logger.info(f"Executing SQL query with {len(params)} parameters")
            
            cursor.execute(sql, params)
            results = cursor.fetchall()
            papers = []

            for row in results:
                try:
                    # Parse authors with better error handling
                    authors = []
                    if row[2]:  # authors field
                        try:
                            author_data = json.loads(row[2])
                            for author in author_data:
                                if isinstance(author, dict):
                                    name_parts = []
                                    if author.get('first_name'):
                                        name_parts.append(author['first_name'])
                                    if author.get('last_name'):
                                        name_parts.append(author['last_name'])
                                    if name_parts:
                                        authors.append(' '.join(name_parts))
                                    elif author.get('last_name'):
                                        authors.append(author['last_name'])
                                elif isinstance(author, str):
                                    authors.append(author)
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse authors for paper {row[0]}")
                            authors = []

                    paper = {
                        'pubmed_id': str(row[0]),  # Ensure PubMed ID is included as string
                        'title': row[1],
                        'authors': authors,  # Use parsed author list
                        'abstract': row[3],
                        'journal': row[4],
                        'year': row[5],
                        'keywords': json.loads(row[6]) if row[6] else [],
                        'relevance_score': row[7],
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{row[0]}/"  # Add PubMed URL
                    }
                    papers.append(paper)
                    logger.info(f"Found paper: {paper['title']} (score: {paper['relevance_score']})")
                    logger.info(f"Authors: {', '.join(paper['authors']) if paper['authors'] else 'No authors listed'}")

                except Exception as e:
                    logger.error(f"Error processing paper {row[0]}: {str(e)}")
                    continue

            # If we don't have enough relevant papers from database, supplement with PubMed
            if len(papers) < limit:
                logger.info(f"Only found {len(papers)} relevant papers in database, fetching more from PubMed")
                pubmed_terms = []
                if search_terms:
                    pubmed_terms.extend(search_terms.get('medical_conditions', []))
                    pubmed_terms.extend(search_terms.get('treatments', []))
                    pubmed_terms.extend(search_terms.get('ethical_principles', []))
                pubmed_papers = await self._search_pubmed(pubmed_terms, limit - len(papers))
                papers.extend(pubmed_papers)

            logger.info(f"Returning {len(papers)} total papers")
            logger.info(f"Relevance scores: {[p['relevance_score'] for p in papers]}")
            return papers

        except Exception as e:
            logger.error(f"Error searching papers in local database: {str(e)}")
            return []

    async def _search_local_db(self, concepts: Dict[str, List[str]], limit: int) -> List[Dict]:
        """Search papers in local database"""
        try:
            # Debug log the concepts
            logger.info(f"Searching with concepts: {concepts}")
            
            # Build SQL query
            sql_query = """
            SELECT DISTINCT 
                p.pubmed_id,
                p.title,
                p.abstract,
                p.journal,
                p.year,
                GROUP_CONCAT(DISTINCT e.consideration) as ethical_considerations
            FROM papers p
            LEFT JOIN ethical_considerations e ON p.pubmed_id = e.paper_id
            WHERE 1=1
            """
            
            params = []
            
            # Build search conditions
            search_conditions = []
            
            # Add ethical principles search
            if concepts['ethical_principles']:
                ethical_terms = concepts['ethical_principles']
                placeholders = ' OR '.join(['p.title LIKE ? OR p.abstract LIKE ?' for _ in ethical_terms])
                search_conditions.append(f"({placeholders})")
                params.extend([f'%{term}%' for term in ethical_terms for _ in range(2)])
            
            # Add medical conditions search
            if concepts['medical_conditions']:
                medical_terms = concepts['medical_conditions']
                placeholders = ' OR '.join(['p.title LIKE ? OR p.abstract LIKE ?' for _ in medical_terms])
                search_conditions.append(f"({placeholders})")
                params.extend([f'%{term}%' for term in medical_terms for _ in range(2)])
            
            # Add stakeholders search
            if concepts['stakeholders']:
                stakeholder_terms = concepts['stakeholders']
                placeholders = ' OR '.join(['p.title LIKE ? OR p.abstract LIKE ?' for _ in stakeholder_terms])
                search_conditions.append(f"({placeholders})")
                params.extend([f'%{term}%' for term in stakeholder_terms for _ in range(2)])
            
            # Add concerns search
            if concepts['concerns']:
                concern_terms = concepts['concerns']
                placeholders = ' OR '.join(['p.title LIKE ? OR p.abstract LIKE ?' for _ in concern_terms])
                search_conditions.append(f"({placeholders})")
                params.extend([f'%{term}%' for term in concern_terms for _ in range(2)])
            
            # Combine all conditions
            if search_conditions:
                sql_query += " AND (" + " OR ".join(search_conditions) + ")"
            
            # Add group by and order
            sql_query += """
            GROUP BY p.pubmed_id, p.title, p.abstract, p.journal, p.year
            ORDER BY p.year DESC
            """
            
            # Debug log the query
            logger.info(f"SQL Query: {sql_query}")
            logger.info(f"Parameters: {params}")
            
            # Execute query
            conn = await self.get_connection()
            cursor = conn.cursor()
            cursor.execute(sql_query, params)
            
            # Fetch and process results
            papers = []
            for row in cursor.fetchall():
                paper = {
                    'pubmed_id': row[0],
                    'title': row[1],
                    'abstract': row[2],
                    'journal': row[3],
                    'year': row[4],
                    'ethical_considerations': row[5].split(',') if row[5] else []
                }
                relevance_score = await self._calculate_relevance_from_concepts(paper, concepts)
                paper['ethical_considerations'] = self._clean_considerations(paper['ethical_considerations'])
                papers.append((paper, relevance_score))
            
            # Debug log the results
            logger.info(f"Found {len(papers)} papers before relevance sorting")
            
            # Sort by relevance score
            papers.sort(key=lambda x: x[1], reverse=True)
            
            # Return top papers
            return [p[0] for p in papers[:limit]]

        except Exception as e:
            logger.error(f"Error searching local database: {str(e)}")
            return []

    async def _search_pubmed(self, concepts: Dict[str, List[str]], limit: int) -> List[Dict]:
        """Search PubMed only when needed"""
        # This method would only be called if specifically requested
        # Implementation remains but is only used when explicitly needed
        return []

    async def get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise

    async def _extract_concepts_from_ai_analysis(self, ai_analysis: Dict) -> Dict[str, List[str]]:
        """Extract search concepts from AI analysis"""
        try:
            # Debug log the input
            logger.info("AI Analysis input:")
            logger.info(f"Summary: {ai_analysis.get('summary', '')[:200]}...")
            logger.info(f"Concerns: {ai_analysis.get('concerns', [])}")
            logger.info(f"Recommendations: {ai_analysis.get('recommendations', [])}")

            prompt = f"""Extract specific search terms from this medical ethics analysis.

Summary: {ai_analysis.get('summary', '')}
Concerns: {', '.join(ai_analysis.get('concerns', []))}
Recommendations: {', '.join(ai_analysis.get('recommendations', []))}

Extract and categorize the most relevant terms into these categories:
1. Ethical principles (e.g., autonomy, beneficence)
2. Medical conditions (e.g., dementia, kidney failure)
3. Stakeholders (e.g., guardian, physician)
4. Concerns (e.g., capacity, consent)

Return EXACTLY in this Python dictionary format:
{{
    'ethical_principles': ['term1', 'term2'],
    'medical_conditions': ['term1', 'term2'],
    'stakeholders': ['term1', 'term2'],
    'concerns': ['term1', 'term2']
}}

Include ONLY the most relevant terms (3-5 per category) that appear in the analysis."""

            response = await self._get_gpt_response(prompt)
            concepts = eval(response.strip())
            
            # Debug log the extracted concepts
            logger.info("Extracted concepts:")
            logger.info(concepts)
            
            return concepts
        except Exception as e:
            logger.error(f"Error extracting concepts: {str(e)}")
            return {
                'ethical_principles': ['autonomy', 'beneficence'],
                'medical_conditions': ['dementia', 'kidney failure'],
                'stakeholders': ['guardian', 'physician'],
                'concerns': ['capacity', 'consent']
            }

    async def _calculate_relevance_from_concepts(self, paper: Dict, concepts: Dict[str, List[str]]) -> float:
        """Calculate relevance score using concepts from AI analysis"""
        try:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            considerations = [c.lower() for c in paper.get('ethical_considerations', [])]
            consideration_count = paper.get('consideration_count', 0)
            
            score = 0
            
            # Base score from ethical considerations count
            score += min(consideration_count / 5, 2)  # Up to 2 points for considerations
            
            # Check ethical principles (highest weight)
            for principle in concepts['ethical_principles']:
                principle = principle.lower()
                if principle in considerations:
                    score += 2
                if principle in title:
                    score += 1.5
                if principle in abstract:
                    score += 1
            
            # Check medical conditions
            for condition in concepts['medical_conditions']:
                condition = condition.lower()
                if condition in title:
                    score += 1.5
                if condition in abstract:
                    score += 1
            
            # Check stakeholders
            for stakeholder in concepts['stakeholders']:
                stakeholder = stakeholder.lower()
                if stakeholder in title:
                    score += 1
                if stakeholder in abstract:
                    score += 0.5
            
            # Check concerns
            for concern in concepts['concerns']:
                concern = concern.lower()
                if concern in considerations:
                    score += 1.5
                if concern in title:
                    score += 1
                if concern in abstract:
                    score += 0.5
            
            # Normalize to 0-10 scale
            return min(score, 10)
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {str(e)}")
            return 0

    async def _calculate_relevance(self, paper: Dict, search_structure: Dict) -> float:
        """Calculate paper relevance score based on query structure"""
        try:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            
            # Build term weights dynamically from search structure
            term_weights = {}
            
            # Add terms from search structure with high weights
            for category, term in search_structure.items():
                term_weights[term.lower()] = 5
            
            # Add general ethical terms with moderate weights
            general_terms = {
                'ethics': 3,
                'ethical': 3,
                'decision making': 3,
                'clinical': 2,
                'healthcare': 2,
                'medical': 2,
                'consultation': 2,
                'autonomy': 2,
                'beneficence': 2,
                'justice': 2,
                'consent': 2
            }
            term_weights.update(general_terms)
            
            score = 0
            # Check title (higher weight)
            for term, weight in term_weights.items():
                if term in title:
                    score += weight * 1.5
            
            # Check abstract
            for term, weight in term_weights.items():
                if term in abstract:
                    score += weight
            
            # Normalize score to 0-10 range
            max_possible_score = sum(w * 2.5 for w in term_weights.values())
            normalized_score = (score / max_possible_score) * 10
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {str(e)}")
            return 0

    async def _analyze_query(self, query: str) -> Dict[str, str]:
        """Extract structured search terms from query"""
        try:
            prompt = f"""Analyze this medical ethics query and extract key search terms.
            Query: {query}
            
            Return a Python dictionary with EXACTLY these keys:
            {
                'condition': 'main medical condition',
                'stakeholder': 'main stakeholder role',
                'setting': 'care setting',
                'ethical_principle': 'main ethical principle'
            }
            
            Use single terms or short phrases. Focus on the most important aspect for each category."""
            
            # Get GPT response
            response = await self._get_gpt_response(prompt)
            # Clean and evaluate response
            cleaned = response.strip().replace('\n', '').replace(' ', '')
            return eval(cleaned)
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return {
                'condition': 'dementia',
                'stakeholder': 'guardian',
                'setting': 'care facility',
                'ethical_principle': 'autonomy'
            }

    def _remove_duplicates(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on PMID"""
        seen_pmids = set()
        unique_papers = []
        for paper in papers:
            pmid = paper.get('pubmed_id')
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                unique_papers.append(paper)
        return unique_papers

    async def debug_database_contents(self):
        """Debug method to check database contents"""
        try:
            conn = await self.get_connection()
            cursor = conn.cursor()
            
            # Check papers table
            cursor.execute("SELECT COUNT(*) FROM papers")
            paper_count = cursor.fetchone()[0]
            logger.info(f"Total papers in database: {paper_count}")
            
            # Sample some papers
            cursor.execute("""
                SELECT p.pubmed_id, p.title, GROUP_CONCAT(e.consideration) as considerations
                FROM papers p
                LEFT JOIN ethical_considerations e ON p.pubmed_id = e.paper_id
                GROUP BY p.pubmed_id
                LIMIT 5
            """)
            sample_papers = cursor.fetchall()
            logger.info("Sample papers in database:")
            for paper in sample_papers:
                logger.info(f"PMID: {paper[0]}")
                logger.info(f"Title: {paper[1]}")
                logger.info(f"Considerations: {paper[2]}")
                logger.info("---")
            
            return paper_count
        except Exception as e:
            logger.error(f"Error checking database: {str(e)}")
            return 0

    async def initialize_database(self):
        """Initialize SQLite database and populate with papers"""
        try:
            # Check if data directory exists
            data_dir = Path('data')
            if not data_dir.exists():
                logger.warning("Creating data directory")
                data_dir.mkdir(parents=True)
            
            papers_file = data_dir / 'pubmed_papers.json'
            logger.info(f"Looking for papers file at: {papers_file.absolute()}")
            
            # Create database schema
            conn = await self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    pubmed_id TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    abstract TEXT, 
                    journal TEXT,
                    year INTEGER,
                    keywords TEXT,
                    relevance_score REAL
                )
            """)
            
            # Load and validate papers
            if papers_file.exists():
                try:
                    with open(papers_file) as f:
                        papers = json.load(f)
                    logger.info(f"Found {len(papers)} papers to load")
                    
                    # Insert papers in batches
                    batch_size = 100
                    for i in range(0, len(papers), batch_size):
                        batch = papers[i:i + batch_size]
                        for paper in batch:
                            cursor.execute("""
                                INSERT OR REPLACE INTO papers 
                                (pubmed_id, title, authors, abstract, journal, year, keywords)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                paper['pubmed_id'],
                                paper['title'],
                                json.dumps(paper.get('authors', [])),
                                paper['abstract'],
                                paper['journal'],
                                paper['year'],
                                json.dumps(paper.get('keywords', []))
                            ))
                    conn.commit()
                    logger.info(f"Successfully loaded {len(papers)} papers into database")
                    
                    # Verify data loaded
                    cursor.execute("SELECT COUNT(*) FROM papers")
                    count = cursor.fetchone()[0]
                    logger.info(f"Database now contains {count} papers")
                    
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in papers file: {papers_file}")
                    return False
                    
            else:
                logger.warning("No papers file found - database will be empty")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            return False

    async def _insert_paper(self, paper: Dict):
        """Insert a paper into the database"""
        try:
            await self.es.index(
                index='medical_ethics',
                document={
                    'title': paper['title'],
                    'authors': paper['authors'],
                    'year': paper['year'],
                    'journal': paper['journal'],
                    'abstract': paper['abstract'],
                    'ethical_considerations': paper['ethical_considerations']
                }
            )
        except Exception as e:
            logger.error(f"Error inserting paper: {str(e)}")
            raise

    async def load_initial_data(self):
        """Load initial data into database from JSON files"""
        try:
            conn = await self.get_connection()
            cursor = conn.cursor()
            
            # Check if database is empty
            cursor.execute("SELECT COUNT(*) FROM papers")
            if cursor.fetchone()[0] == 0:
                logger.info("Database is empty, loading data from JSON files...")
                
                # Load data from JSON files in data directory
                data_dir = Path('data')
                json_files = list(data_dir.glob('pubmed_ethics_papers*.json'))
                
                if not json_files:
                    logger.error("No JSON files found in data directory")
                    return False
                    
                # Keep track of seen PMIDs to avoid duplicates
                seen_pmids = set()
                total_papers = 0
                
                for json_file in json_files:
                    logger.info(f"Loading data from {json_file}")
                    try:
                        with open(json_file, 'r') as f:
                            papers = json.load(f)
                            
                        for paper in papers:
                            pmid = paper.get('pubmed_id')
                            
                            # Skip if we've already seen this PMID
                            if pmid in seen_pmids:
                                continue
                            seen_pmids.add(pmid)
                            
                            # Extract ethical considerations from title and abstract
                            ethical_terms = set()
                            title = paper.get('title', '').lower()
                            abstract = paper.get('abstract', '').lower()
                            
                            # Extract from title (higher priority)
                            title_terms = self._extract_ethical_terms(title, self.ethics_keywords)
                            ethical_terms.update(title_terms)
                            
                            # Extract from abstract
                            abstract_terms = self._extract_ethical_terms(abstract, self.ethics_keywords)
                            ethical_terms.update(abstract_terms)
                            
                            # Insert paper
                            cursor.execute("""
                                INSERT OR IGNORE INTO papers 
                                (pubmed_id, title, abstract, journal, year, authors, keywords, relevance_score)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                pmid,
                                paper.get('title'),
                                paper.get('abstract'),
                                paper.get('journal'),
                                paper.get('year'),
                                json.dumps(paper.get('authors', [])),
                                json.dumps(paper.get('keywords', [])),
                                paper.get('relevance_score', 0)
                            ))
                            
                            # Insert extracted ethical considerations
                            for consideration in ethical_terms:
                                cursor.execute("""
                                    INSERT INTO ethical_considerations 
                                    (paper_id, consideration)
                                    VALUES (?, ?)
                                """, (pmid, consideration))
                            
                            total_papers += 1
                            
                            # Commit every 1000 papers
                            if total_papers % 1000 == 0:
                                conn.commit()
                                logger.info(f"Processed {total_papers} unique papers...")
                    
                        conn.commit()
                        logger.info(f"Successfully loaded papers from {json_file}")
                            
                    except Exception as e:
                        logger.error(f"Error loading file {json_file}: {str(e)}")
                        continue
                
                logger.info(f"Total unique papers loaded into database: {total_papers}")
                logger.info(f"Duplicate papers skipped: {len(seen_pmids) - total_papers}")
                
                # Verify data was loaded
                cursor.execute("SELECT COUNT(*) FROM papers")
                paper_count = cursor.fetchone()[0]
                logger.info(f"Database now contains {paper_count} papers")
                
                cursor.execute("SELECT COUNT(*) FROM ethical_considerations")
                considerations_count = cursor.fetchone()[0]
                logger.info(f"Database contains {considerations_count} ethical considerations")
                
                # Sample the data to verify
                cursor.execute("""
                    SELECT p.title, GROUP_CONCAT(e.consideration) as considerations
                    FROM papers p
                    LEFT JOIN ethical_considerations e ON p.pubmed_id = e.paper_id
                    GROUP BY p.pubmed_id
                    LIMIT 5
                """)
                logger.info("Sample papers from loaded data:")
                for row in cursor.fetchall():
                    logger.info(f"Title: {row[0]}")
                    logger.info(f"Considerations: {row[1] or 'None'}")
                    logger.info("---")
                
                return True
            return True
        except Exception as e:
            logger.error(f"Error loading initial data: {str(e)}")
            return False

    async def verify_database(self):
        """Verify database is properly initialized"""
        try:
            conn = await self.get_connection()
            cursor = conn.cursor()
            
            # Check tables exist and have data
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            logger.info(f"Database tables: {[t[0] for t in tables]}")
            
            # Check papers count
            cursor.execute("SELECT COUNT(*) FROM papers")
            paper_count = cursor.fetchone()[0]
            logger.info(f"Papers in database: {paper_count}")
            
            # Check ethical considerations
            cursor.execute("SELECT COUNT(*) FROM ethical_considerations")
            considerations_count = cursor.fetchone()[0]
            logger.info(f"Ethical considerations in database: {considerations_count}")
            
            # Sample a paper to verify content
            cursor.execute("""
                SELECT p.*, GROUP_CONCAT(e.consideration) as considerations
                FROM papers p
                LEFT JOIN ethical_considerations e ON p.pubmed_id = e.paper_id
                GROUP BY p.pubmed_id
                LIMIT 1
            """)
            sample = cursor.fetchone()
            if sample:
                logger.info("Sample paper from database:")
                logger.info(f"Title: {sample['title']}")
                logger.info(f"Considerations: {sample['considerations']}")
            
            return paper_count > 0 and considerations_count > 0
            
        except Exception as e:
            logger.error(f"Error verifying database: {str(e)}")
            return False

    def _extract_ethical_terms(self, text: str, keywords: list) -> set:
        """Extract ethical terms with context awareness"""
        ethical_terms = set()
        text = text.lower()
        
        # Direct keyword matching
        for term in keywords:
            if term in text:
                ethical_terms.add(term)
        
        # Extract title-specific terms
        if "ethic" in text or "moral" in text:
            ethical_terms.add("ethics")
            # Look for specific types of ethics
            if "committee" in text:
                ethical_terms.add("ethics committee")
            if "research" in text:
                ethical_terms.add("research ethics")
            if "clinical" in text:
                ethical_terms.add("clinical ethics")
        
        # Look for phrases containing ethical terms
        sentences = text.split('.')
        for sentence in sentences:
            # If sentence contains ethical keywords, extract relevant phrases
            if any(term in sentence for term in ['ethic', 'moral', 'consent', 'right', 'justice', 'autonomy']):
                words = sentence.split()
                # Look at 2-4 word phrases
                for n in range(2, 5):
                    for i in range(len(words)-n+1):
                        phrase = ' '.join(words[i:i+n]).strip()
                        # Check if phrase contains any ethical terms
                        if any(term in phrase for term in keywords):
                            # Clean up the phrase
                            phrase = phrase.strip('.,()[]{}').strip()
                            # Filter out low-quality phrases
                            if (len(phrase) > 3 and  # Avoid too-short phrases
                                not phrase.startswith(('and ', 'the ', 'for ')) and  # Avoid fragments
                                not any(word in phrase for word in ['is', 'are', 'was', 'were']) and  # Avoid verb fragments
                                len(phrase.split()) >= 2):  # Ensure meaningful phrases
                                ethical_terms.add(phrase)
        
        # Add domain-specific ethical considerations
        domain_terms = {
            'vaccine': ['medical autonomy', 'mandatory vaccination', 'informed consent'],
            'military': ['duty', 'service ethics', 'military ethics'],
            'dna': ['genetic privacy', 'data protection', 'informed consent'],
            'clinical trial': ['research ethics', 'participant rights', 'informed consent'],
            'treatment': ['patient autonomy', 'medical ethics', 'treatment ethics']
        }
        
        for domain, terms in domain_terms.items():
            if domain in text:
                ethical_terms.update(terms)
        
        # Add related terms for common ethical concepts
        related_terms = {
            'consent': ['informed consent', 'patient consent', 'consent process'],
            'ethics committee': ['institutional review', 'ethics board', 'ethical review'],
            'autonomy': ['patient autonomy', 'personal autonomy', 'autonomous decision'],
            'justice': ['social justice', 'healthcare justice', 'equitable access'],
            'privacy': ['data privacy', 'confidentiality', 'information protection']
        }
        
        for term in ethical_terms.copy():
            if term in related_terms:
                for related in related_terms[term]:
                    if related in text:
                        ethical_terms.add(related)
        
        return ethical_terms 

    async def cleanup_database(self, min_considerations: int = None):
        """Remove papers with insufficient ethical considerations
        
        Args:
            min_considerations (int, optional): Minimum number of ethical considerations required.
                Defaults to self.MIN_ETHICAL_CONSIDERATIONS (2)
        """
        if min_considerations is None:
            min_considerations = self.MIN_ETHICAL_CONSIDERATIONS
        try:
            conn = await self.get_connection()
            cursor = conn.cursor()
            
            # Get papers with few ethical considerations
            cursor.execute("""
                WITH paper_considerations AS (
                    SELECT paper_id, COUNT(DISTINCT consideration) as consideration_count
                    FROM ethical_considerations
                    GROUP BY paper_id
                )
                DELETE FROM papers
                WHERE pubmed_id IN (
                    SELECT p.pubmed_id
                    FROM papers p
                    LEFT JOIN paper_considerations pc ON p.pubmed_id = pc.paper_id
                    WHERE pc.consideration_count < ? OR pc.consideration_count IS NULL
                )
            """, (min_considerations,))
            
            # Also clean up orphaned considerations
            cursor.execute("""
                DELETE FROM ethical_considerations
                WHERE paper_id NOT IN (SELECT pubmed_id FROM papers)
            """)
            
            conn.commit()
            
            # Get new counts
            cursor.execute("SELECT COUNT(*) FROM papers")
            paper_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM ethical_considerations")
            considerations_count = cursor.fetchone()[0]
            
            logger.info(f"After cleanup: {paper_count} papers with {considerations_count} ethical considerations")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up database: {str(e)}")
            return False 

    def _clean_considerations(self, considerations: List[str]) -> List[str]:
        """Clean and deduplicate ethical considerations"""
        cleaned = set()
        for consideration in considerations:
            # Remove HTML tags and special characters
            consideration = consideration.replace('<b>', '').replace('</b>', '')
            consideration = consideration.replace('<i>', '').replace('</i>', '')
            consideration = consideration.replace('"', '').replace("'", "")
            consideration = consideration.strip('.,()[]{}:').strip()
            
            # Skip if too short or contains unwanted patterns
            if (len(consideration.split()) < 2 or  # Too short
                consideration.startswith(('and ', 'the ', 'for ', 'to ', 'of ', 'in ')) or  # Fragments
                any(word in consideration for word in ['is', 'are', 'was', 'were']) or  # Verb fragments
                consideration.endswith(' and') or  # Incomplete phrases
                consideration.isdigit() or  # Numbers
                len(consideration) < 8):  # Too short strings
                continue
            
            # Normalize spaces
            consideration = ' '.join(consideration.split())
            
            # Check for duplicates and substrings
            if any(consideration in existing for existing in cleaned):
                continue
            if any(existing in consideration for existing in cleaned):
                continue
            
            cleaned.add(consideration)
        
        # Sort by length and alphabetically
        return sorted(list(cleaned), key=lambda x: (-len(x), x)) 

    async def initialize_elasticsearch(self):
        """Initialize Elasticsearch index with proper mappings"""
        try:
            # Define index mapping
            mapping = {
                "mappings": {
                    "properties": {
                        "title": {
                            "type": "text",
                            "analyzer": "english"
                        },
                        "abstract": {
                            "type": "text",
                            "analyzer": "english"
                        },
                        "journal": {"type": "keyword"},
                        "year": {"type": "keyword"},  # Changed from integer to keyword
                        "authors": {
                            "type": "nested",  # Changed from keyword to nested
                            "properties": {
                                "last_name": {"type": "keyword"}
                            }
                        },
                        "ethical_considerations": {
                            "type": "text",
                            "analyzer": "english"
                        }
                    }
                }
            }

            # Delete existing index if it exists
            if await self.es.indices.exists(index=self.index_name):
                await self.es.indices.delete(index=self.index_name)
                logger.info(f"Deleted existing index: {self.index_name}")

            # Create new index
            await self.es.indices.create(
                index=self.index_name,
                body=mapping
            )
            logger.info(f"Created Elasticsearch index: {self.index_name}")

            return True

        except Exception as e:
            logger.error(f"Error initializing Elasticsearch: {str(e)}")
            return False

    async def index_exists(self):
        """Check if the index already exists"""
        try:
            exists = await self.es.indices.exists(index=self.index_name)
            return exists
        except Exception as e:
            logger.error(f"Error checking index existence: {str(e)}")
            return False

    async def migrate_to_elasticsearch(self):
        """Migrate data from SQLite to Elasticsearch"""
        try:
            # Check if index already exists and has data
            if await self.index_exists():
                logger.info(f"Index {self.index_name} already exists, skipping migration")
                return True

            # If index doesn't exist, proceed with migration
            logger.info("Starting migration from SQLite to Elasticsearch...")
            
            # First initialize Elasticsearch index
            await self.initialize_elasticsearch()
            
            # Get SQLite connection
            conn = await self.get_connection()
            cursor = conn.cursor()
            
            # Get all papers with their ethical considerations
            cursor.execute("""
                SELECT 
                    p.pubmed_id,
                    p.title,
                    p.abstract,
                    p.journal,
                    p.year,
                    p.authors,
                    GROUP_CONCAT(e.consideration) as ethical_considerations
                FROM papers p
                LEFT JOIN ethical_considerations e ON p.pubmed_id = e.paper_id
                GROUP BY p.pubmed_id
            """)
            
            # Migrate in batches
            batch_size = 100
            batch = []
            total_migrated = 0
            
            for row in cursor:
                doc = {
                    'pubmed_id': row[0],
                    'title': row[1],
                    'abstract': row[2],
                    'journal': row[3],
                    'year': row[4],
                    'authors': json.loads(row[5]) if row[5] else [],
                    'ethical_considerations': row[6].split(',') if row[6] else []
                }
                batch.append(doc)
                
                if len(batch) >= batch_size:
                    # Index the batch
                    body = []
                    for doc in batch:
                        body.extend([
                            {'index': {'_index': self.index_name}},
                            doc
                        ])
                    await self.es.bulk(body=body)
                    
                    total_migrated += len(batch)
                    logger.info(f"Migrated {total_migrated} papers...")
                    batch = []
            
            # Index any remaining documents
            if batch:
                body = []
                for doc in batch:
                    body.extend([
                        {'index': {'_index': self.index_name}},
                        doc
                    ])
                await self.es.bulk(body=body)
                total_migrated += len(batch)
            
            logger.info(f"Migration complete. Total papers migrated: {total_migrated}")
            return True
            
        except Exception as e:
            logger.error(f"Error during migration: {str(e)}")
            return False 

    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.es.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup() 

    async def reset_elasticsearch(self):
        """Reset Elasticsearch index"""
        try:
            # Delete index if exists
            if await self.es.indices.exists(index=self.index_name):
                await self.es.indices.delete(index=self.index_name)
                logger.info(f"Deleted index: {self.index_name}")
            
            # Initialize new index
            await self.initialize_elasticsearch()
            
            # Migrate data
            await self.migrate_to_elasticsearch()
            
            return True
        except Exception as e:
            logger.error(f"Error resetting Elasticsearch: {str(e)}")
            return False 