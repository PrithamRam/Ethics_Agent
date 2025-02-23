import asyncio
import aiosqlite
import pandas as pd
import os

async def populate_db():
    # Read the CSV file with ethics references
    csv_path = "csv-ethic-set.csv"  # Using the provided CSV file
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} references")
    
    async with aiosqlite.connect("ethics.db") as db:
        # Create table matching the CSV structure
        await db.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                PMID TEXT PRIMARY KEY,
                Title TEXT,
                Authors TEXT,
                Citation TEXT,
                First_Author TEXT,
                Journal TEXT,
                Year TEXT,
                Create_Date TEXT,
                PMCID TEXT,
                NIHMSID TEXT,
                DOI TEXT
            )
        """)
        
        # Insert data from CSV in batches
        count = 0
        for _, row in df.iterrows():
            try:
                await db.execute("""
                    INSERT OR REPLACE INTO papers 
                    (PMID, Title, Authors, Citation, First_Author, Journal, 
                     Year, Create_Date, PMCID, NIHMSID, DOI)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(row['PMID']),
                    row['Title'],
                    row['Authors'],
                    row['Citation'],
                    row['First Author'],
                    row['Journal/Book Publication'],
                    row['Year'],
                    row['Create Date'],
                    row.get('PMCID', ''),
                    row.get('NIHMSID', ''),
                    row.get('DOI', '')
                ))
                count += 1
                if count % 1000 == 0:
                    print(f"Imported {count} references...")
            except Exception as e:
                print(f"Error importing row: {e}")
                continue
        
        await db.commit()
        print(f"Successfully imported {count} references")

if __name__ == "__main__":
    asyncio.run(populate_db()) 