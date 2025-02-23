from typing import List, Dict, Optional, Tuple
import aiohttp
from Bio import Entrez, Medline
import asyncio
from src.config import SystemConfig
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class PubMedHandler:
    def __init__(self, config: SystemConfig):
        self.config = config
        # Use new config structure
        Entrez.email = config.pubmed_email
        self.max_results = config.max_results
        
        if config.pubmed_api_key:
            Entrez.api_key = config.pubmed_api_key

        self.cache = {}  # Simple cache for paper details
        
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
                fetched_papers = await self._fetch_from_pubmed(missing_ids)
                # Update cache with new papers
                for paper in fetched_papers:
                    self.cache[paper['pubmed_id']] = paper
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