import asyncio
import logging
from src.medical_ethics_assistant import MedicalEthicsAssistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_search():
    # Initialize the assistant
    assistant = await MedicalEthicsAssistant.create()
    
    # Test cases
    test_queries = [
        {
            "query": "What are the ethical considerations for mandatory vaccination in healthcare workers?",
            "ai_analysis": {
                "summary": "This query concerns the balance between public health safety and individual autonomy in healthcare settings. Key considerations include mandatory vaccination policies, healthcare worker rights, and patient protection.",
                "concerns": [
                    "Individual autonomy vs collective safety",
                    "Professional obligations of healthcare workers",
                    "Vaccine mandates and personal choice"
                ],
                "recommendations": [
                    "Consider both individual rights and public health",
                    "Review existing vaccination policies",
                    "Evaluate ethical frameworks for mandatory medical interventions"
                ]
            }
        },
        {
            "query": "What are the ethical implications of genetic testing in prenatal care?",
            "ai_analysis": {
                "summary": "This query explores ethical challenges in prenatal genetic testing, including informed consent, privacy, and decision-making autonomy.",
                "concerns": [
                    "Right to genetic privacy",
                    "Informed consent in prenatal testing",
                    "Ethical use of genetic information"
                ],
                "recommendations": [
                    "Develop comprehensive consent protocols",
                    "Protect genetic privacy",
                    "Provide genetic counseling support"
                ]
            }
        }
    ]
    
    # Run tests
    for test_case in test_queries:
        logger.info(f"\nTesting query: {test_case['query']}")
        logger.info("=" * 80)
        
        papers = await assistant.get_relevant_papers(
            query=test_case['query'],
            ai_analysis=test_case['ai_analysis']
        )
        
        logger.info(f"\nFound {len(papers)} relevant papers")
        logger.info("-" * 40)
        
        for i, paper in enumerate(papers, 1):
            logger.info(f"\nPaper {i}:")
            logger.info(f"Title: {paper['title']}")
            logger.info(f"Year: {paper.get('year', 'N/A')}")
            logger.info(f"Ethical considerations: {', '.join(paper['ethical_considerations'])}")
            logger.info(f"Consideration count: {paper['consideration_count']}")
            logger.info("-" * 40)

if __name__ == "__main__":
    asyncio.run(test_search()) 