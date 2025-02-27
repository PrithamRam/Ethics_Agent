import asyncio
from src.ethics_db import EthicsDB

async def test_connection():
    db = EthicsDB()
    try:
        # Test if we can initialize the index
        success = await db.initialize_elasticsearch()
        if success:
            print("Successfully connected to Elasticsearch and initialized index!")
        else:
            print("Failed to initialize Elasticsearch index")
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")
    finally:
        # Close the Elasticsearch client
        await db.es.close()

if __name__ == "__main__":
    asyncio.run(test_connection()) 