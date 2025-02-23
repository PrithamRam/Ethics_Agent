from typing import List, Dict, Optional, Tuple
import aiohttp
from Bio import Entrez, Medline
import asyncio
from src.config import SystemConfig
import logging
import xml.etree.ElementTree as ET
import json
import time
from pathlib import Path
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class PubMedHandler:
    def __init__(self):
        load_dotenv()
        # Debug: Print all environment variables
        logger.info("Environment variables:")
        logger.info(f"PUBMED_EMAIL: {os.getenv('PUBMED_EMAIL')}")
        logger.info(f"ENTREZ_EMAIL: {os.getenv('ENTREZ_EMAIL')}")
        logger.info(f"PUBMED_API_KEY: {os.getenv('PUBMED_API_KEY')}")
        logger.info(f"ENTREZ_API_KEY: {os.getenv('ENTREZ_API_KEY')}")
        
        # Try both PUBMED_EMAIL and ENTREZ_EMAIL
        self.email = os.getenv('ENTREZ_EMAIL') or os.getenv('PUBMED_EMAIL')
        if not self.email:
            raise ValueError("Neither ENTREZ_EMAIL nor PUBMED_EMAIL set in .env file")
        Entrez.email = self.email
        
        # Try both API key variants
        self.api_key = os.getenv('ENTREZ_API_KEY') or os.getenv('PUBMED_API_KEY')
        if self.api_key:
            Entrez.api_key = self.api_key
            
        logger.info(f"Initialized PubMed handler with email: {self.email}")

    async def fetch_papers(self, pubmed_ids: List[str], max_papers: int = 10) -> List[Dict]:
        """Fetch paper details from PubMed"""
        if not pubmed_ids:
            return []
            
        papers = []
        unique_ids = list(set(pubmed_ids))[:max_papers]
        
        # Check cache first
        cached_papers = [self.cache.get(pmid) for pmid in unique_ids]
        missing_ids = [pmid for pmid, paper in zip(unique_ids, cached_papers) if paper is None]
        
        if missing_ids:
            try:
                # Add delay to avoid rate limits
                await asyncio.sleep(1)  # 1 second delay between requests
                
                fetched_papers = await self._fetch_from_pubmed(missing_ids)
                # Update cache with new papers
                for paper in fetched_papers:
                    self.cache[paper['pubmed_id']] = paper
                    # Initialize empty ethical_considerations if not present
                    if 'ethical_considerations' not in paper:
                        paper['ethical_considerations'] = []
                papers.extend(fetched_papers)
            except Exception as e:
                logging.error(f"Error fetching papers from PubMed: {str(e)}")
        
        # Add cached papers
        papers.extend([p for p in cached_papers if p is not None])
        return papers
    
    async def _fetch_from_pubmed(self, pubmed_ids: List[str]) -> List[Dict]:
        """Fetch papers from PubMed API"""
        async with aiohttp.ClientSession() as session:
            papers = []
            for pmid in pubmed_ids:
                try:
                    paper = await self._fetch_single_paper(session, pmid)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logging.error(f"Error fetching paper {pmid}: {str(e)}")
            return papers
    
    async def _fetch_single_paper(self, session: aiohttp.ClientSession, pmid: str) -> Optional[Dict]:
        """Fetch a single paper from PubMed"""
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'xml'
        }
        
        try:
            response = await session.get(url, params=params)
            if response.status == 200:
                xml_content = await response.text()
                logging.debug(f"Received XML content: {xml_content[:200]}...")
                return self._parse_pubmed_xml(xml_content, pmid)
            logging.error(f"Failed to fetch paper {pmid}: Status {response.status}")
            return None
        except Exception as e:
            logging.error(f"Error fetching paper {pmid}: {str(e)}")
            return None
    
    def _parse_pubmed_xml(self, xml_content: str, pmid: str) -> Optional[Dict]:
        """Parse PubMed XML response"""
        try:
            root = ET.fromstring(xml_content)
            article = root.find('.//Article')
            if article is None:
                logging.error(f"No article found in XML for {pmid}")
                return None
            
            title = article.findtext('.//ArticleTitle', '')
            abstract = article.findtext('.//Abstract/AbstractText', '')
            
            # Extract authors
            authors = []
            for author in article.findall('.//Author'):
                authors.append({
                    'last_name': author.findtext('LastName', ''),
                    'fore_name': author.findtext('ForeName', '')
                })
            
            # Extract keywords
            keywords = [k.text for k in article.findall('.//Keyword') if k.text]
            
            return {
                'pubmed_id': pmid,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'keywords': keywords
            }
        except ET.ParseError as e:
            logging.error(f"XML parsing error for {pmid}: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Error parsing paper {pmid}: {str(e)}")
            return None
    
    def _extract_publication_date(self, article: ET.Element) -> str:
        """Extract publication date from article XML"""
        pub_date = article.find('.//PubDate')
        if pub_date is not None:
            year = pub_date.find('Year')
            month = pub_date.find('Month')
            day = pub_date.find('Day')
            
            date_parts = []
            if year is not None:
                date_parts.append(year.text)
            if month is not None:
                date_parts.append(month.text)
            if day is not None:
                date_parts.append(day.text)
                
            return '-'.join(date_parts)
        return ''
    
    def _extract_authors(self, article: ET.Element) -> List[Dict]:
        """Extract authors from article XML"""
        authors = []
        author_list = article.findall('.//Author')
        
        for author in author_list:
            last_name = author.find('LastName')
            fore_name = author.find('ForeName')
            authors.append({
                'last_name': last_name.text if last_name is not None else '',
                'fore_name': fore_name.text if fore_name is not None else ''
            })
        
        return authors

    async def search_pubmed(self, query: str) -> Tuple[List[str], int]:
        """Search PubMed for relevant papers"""
        try:
            # Add ethics-related terms to query
            ethics_query = f"{query} AND (ethics OR ethical OR bioethics OR moral)"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params={
                        "db": "pubmed",
                        "term": ethics_query,
                        "retmode": "json",
                        "retmax": self.max_results,
                        "sort": "relevance"
                    }
                ) as response:
                    if response.status != 200:
                        raise Exception(f"PubMed API error: {response.status}")
                    
                    data = await response.json()
                    
                    if "esearchresult" not in data:
                        raise Exception("Invalid response from PubMed")
                        
                    pmids = data["esearchresult"].get("idlist", [])
                    total_results = int(data["esearchresult"].get("count", 0))
                    
                    return pmids, total_results

        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            return [], 0

    async def get_relevant_papers(self, query: str) -> List[Dict]:
        """Get relevant papers for a query"""
        pmids, _ = await self.search_pubmed(query)
        return await self.fetch_papers(pmids)

    async def get_paper_details(self, pubmed_id: str) -> Dict:
        """Get detailed paper information from PubMed"""
        try:
            papers = await self.fetch_papers([pubmed_id])
            if papers:
                paper = papers[0]
                # Add PubMed URL
                paper['pubmed_url'] = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
                # Format citation
                authors = paper.get('authors', [])
                if authors:
                    first_author = f"{authors[0]['last_name']} {authors[0]['fore_name']}"
                    paper['citation'] = f"{first_author} et al. PMID: {pubmed_id}"
                else:
                    paper['citation'] = f"PMID: {pubmed_id}"
                return paper
            return None
        except Exception as e:
            logger.error(f"Error getting paper details: {str(e)}")
            return None

    def search_ethics_papers(self, max_results: int = 100000, start_from: int = 10000) -> List[Dict]:
        """Search PubMed for ethics-related papers from 1999 to present"""
        try:
            # Load existing papers
            existing_papers = []
            existing_file = Path('data') / 'pubmed_ethics_papers_error_recovery.json'
            if existing_file.exists():
                with open(existing_file, 'r') as f:
                    existing_papers = json.load(f)
                    logger.info(f"Loaded {len(existing_papers)} existing papers")
            
            papers = existing_papers
            chunk_size = 100
            
            # Define years to search (from most recent to oldest)
            years = list(range(2024, 1998, -1))
            
            base_query = """
                (ethics[Title/Abstract] OR ethical[Title/Abstract] OR bioethics[Title/Abstract] OR 
                moral[Title/Abstract] OR "human rights"[Title/Abstract]) AND 
                (medical[Title/Abstract] OR clinical[Title/Abstract] OR healthcare[Title/Abstract] OR 
                "health care"[Title/Abstract] OR medicine[Title/Abstract])
            """
            
            for year in years:
                query = f"{base_query} AND {year}[Date - Publication]"
                logger.info(f"\nSearching papers from {year}...")
                
                handle = Entrez.esearch(
                    db="pubmed",
                    term=query,
                    retmax=0,
                    usehistory='y',
                    api_key=self.api_key
                )
                results = Entrez.read(handle)
                handle.close()
                
                total_count = int(results["Count"])
                logger.info(f"Found {total_count} papers for {year}")
                
                if total_count == 0:
                    continue
                    
                webenv = results["WebEnv"]
                query_key = results["QueryKey"]
                
                # Process papers in chunks
                for start in range(0, total_count, chunk_size):
                    try:
                        handle = Entrez.efetch(
                            db="pubmed",
                            retstart=start,
                            retmax=chunk_size,
                            webenv=webenv,
                            query_key=query_key,
                            rettype="medline",
                            retmode="xml",
                            api_key=self.api_key
                        )
                        
                        records = Entrez.read(handle)
                        handle.close()
                        
                        for record in records['PubmedArticle']:
                            try:
                                article = record['MedlineCitation']['Article']
                                pub_year = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {}).get('Year', '')
                                
                                if pub_year and int(pub_year) == year:  # Only exact year matches
                                    # Check if we already have this paper
                                    pmid = record['MedlineCitation']['PMID']
                                    if not any(p.get('pubmed_id') == pmid for p in existing_papers):
                                        paper = {
                                            'pubmed_id': pmid,
                                            'title': article.get('ArticleTitle', ''),
                                            'abstract': article.get('Abstract', {}).get('AbstractText', [''])[0],
                                            'keywords': [kw.lower() for kw in article.get('KeywordList', [[]])[0]],
                                            'ethical_considerations': self._extract_ethical_terms(
                                                article.get('ArticleTitle', '') + ' ' + 
                                                article.get('Abstract', {}).get('AbstractText', [''])[0]
                                            ),
                                            'journal': article.get('Journal', {}).get('Title', ''),
                                            'year': pub_year
                                        }
                                        papers.append(paper)
                                        logger.info(f"Added paper from {pub_year}: {paper['title'][:100]}...")
                            
                            except Exception as e:
                                logger.error(f"Error processing paper {record.get('MedlineCitation', {}).get('PMID', '')}: {str(e)}")
                                continue
                        
                        # Save intermediate results every 100 new papers
                        if (len(papers) - len(existing_papers)) % 100 == 0:
                            output_file = Path('data') / f'pubmed_ethics_papers_intermediate_{len(papers)}.json'
                            with open(output_file, 'w') as f:
                                json.dump(papers, f, indent=2)
                            logger.info(f"Saved intermediate results ({len(papers)} papers) to {output_file}")
                        
                        time.sleep(0.5)  # Reduced delay between chunks
                        
                    except Exception as e:
                        logger.error(f"Error fetching chunk: {str(e)}")
                        # Save on error
                        output_file = Path('data') / 'pubmed_ethics_papers_error_recovery.json'
                        with open(output_file, 'w') as f:
                            json.dump(papers, f, indent=2)
                        logger.info(f"Saved {len(papers)} papers to error recovery file after error")
                        continue
                    
                time.sleep(1)  # Delay between years
                
                # Save after each year
                output_file = Path('data') / f'pubmed_ethics_papers_{year}.json'
                with open(output_file, 'w') as f:
                    json.dump(papers, f, indent=2)
                logger.info(f"Completed year {year}. Total papers so far: {len(papers)}")
                
            return papers
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            # Save what we have so far
            if papers:
                output_file = Path('data') / 'pubmed_ethics_papers_error_recovery.json'
                with open(output_file, 'w') as f:
                    json.dump(papers, f, indent=2)
                logger.info(f"Saved {len(papers)} papers to error recovery file")
            return []
    
    def _extract_ethical_terms(self, text: str) -> List[str]:
        """Extract ethical considerations from text"""
        ethical_terms = {
            'informed consent', 'autonomy', 'beneficence', 'nonmaleficence',
            'justice', 'privacy', 'confidentiality', 'human rights',
            'dignity', 'equity', 'fairness', 'discrimination',
            'resource allocation', 'ethical dilemma', 'moral obligation',
            'professional ethics', 'research ethics', 'ethical principles'
        }
        
        found_terms = []
        text_lower = text.lower()
        for term in ethical_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms

    def _save_to_json(self, papers: List[Dict], filename: str) -> None:
        """Save papers to JSON file with error handling"""
        try:
            output_file = Path('data') / filename
            logger.info(f"Saving to {output_file.absolute()}")
            with open(output_file, 'w') as f:
                json.dump(papers, f, indent=2)
            logger.info(f"Successfully saved {len(papers)} papers to {output_file}")
        except Exception as e:
            logger.error(f"Error saving to {filename}: {str(e)}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Debug: Print current directory and .env location
    current_dir = os.getcwd()
    env_path = os.path.join(current_dir, '.env')
    logger.info(f"Current directory: {current_dir}")
    logger.info(f"Looking for .env at: {env_path}")
    logger.info(f"Does .env exist? {os.path.exists(env_path)}")
    
    # Create data directory if needed
    data_dir = Path('data')
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created data directory at {data_dir.absolute()}")
    
    # Fetch papers starting from 10000
    handler = PubMedHandler()
    papers = handler.search_ethics_papers(start_from=10000)
    
    # Save to JSON file
    output_file = data_dir / 'pubmed_ethics_papers.json'
    with open(output_file, 'w') as f:
        json.dump(papers, f, indent=2)
    
    logger.info(f"Saved {len(papers)} papers to {output_file}") 