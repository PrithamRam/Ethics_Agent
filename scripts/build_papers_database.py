import json
import logging
from pathlib import Path
from Bio import Entrez
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_ethics_papers():
    """Fetch medical ethics papers from PubMed"""
    
    # Search query for medical ethics papers
    query = """("medical ethics"[MeSH Terms] OR "bioethics"[MeSH Terms]) AND 
              ("2010"[Date - Publication] : "2023"[Date - Publication]) AND 
              (("cultural competency"[MeSH Terms]) OR 
               ("informed consent"[MeSH Terms]) OR
               ("patient rights"[MeSH Terms]) OR
               ("medical ethics"[Title/Abstract]) OR
               ("ethical issues"[Title/Abstract]))"""
               
    try:
        # Search PubMed
        handle = Entrez.esearch(db="pubmed", term=query, retmax=1000)
        record = Entrez.read(handle)
        handle.close()
        
        pmids = record["IdList"]
        logger.info(f"Found {len(pmids)} papers")
        
        papers = []
        batch_size = 100
        
        # Fetch papers in batches
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            handle = Entrez.efetch(db="pubmed", id=batch_pmids, rettype="medline", retmode="text")
            records = Medline.parse(handle)
            
            for record in records:
                paper = {
                    'pubmed_id': record.get('PMID', ''),
                    'title': record.get('TI', ''),
                    'authors': parse_authors(record.get('AU', [])),
                    'abstract': record.get('AB', ''),
                    'journal': record.get('JT', ''),
                    'year': int(record.get('DP', '0')[:4]),
                    'keywords': record.get('MH', [])
                }
                papers.append(paper)
                
            logger.info(f"Fetched {len(papers)} papers so far...")
            time.sleep(1)  # Be nice to NCBI
            
        return papers
        
    except Exception as e:
        logger.error(f"Error fetching papers: {e}")
        return []

def parse_authors(author_list):
    """Parse author names into structured format"""
    authors = []
    for author in author_list:
        parts = author.split()
        if len(parts) > 1:
            authors.append({
                'last_name': parts[-1],
                'first_name': ' '.join(parts[:-1])
            })
        else:
            authors.append({'last_name': author})
    return authors

def main():
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Fetch papers
    papers = fetch_ethics_papers()
    
    if papers:
        # Save to JSON
        output_file = data_dir / 'pubmed_papers.json'
        with open(output_file, 'w') as f:
            json.dump(papers, f, indent=2)
        logger.info(f"Saved {len(papers)} papers to {output_file}")
    else:
        logger.error("No papers fetched")

if __name__ == "__main__":
    main() 