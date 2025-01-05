from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os
from datetime import datetime
import re
import pandas as pd
import argparse

load_dotenv()


def init_qdrant_client():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_KEY")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    return client


def count_data(collection_name, client):
    source_name = collection_name
    collection_info = client.get_collection(collection_name=source_name)
    points_count = collection_info.points_count
    print(f"Total points in the collection: {points_count}")
    return points_count


def extract_payload(collection_name, points_count, client):
    records = []
    result = client.scroll(
        collection_name=collection_name,
        limit=points_count,  # Adjust limit if needed
        with_payload=True
    )
    records.append(result)
    return records


def extract_source(records):
    source_urls = []
    for record_group, _ in records:
        for rec in record_group:
            if 'metadata' in rec.payload:
                # Find keys containing 'date' (case-insensitive)
                source_keys = [k for k in rec.payload['metadata'].keys() if 'source' in k.lower()]
                for sources in source_keys:
                    source_value = rec.payload['metadata'].get('source')
                    if source_value:
                        # Store each date with its corresponding point ID
                        source_urls.append((source_value))

    return source_urls  # Print only the dates for clarity


def main(collection_name, output_path):
    """Main function to process the collection and save results."""
    client = init_qdrant_client()
    points_count = count_data(collection_name, client)
    records = extract_payload(collection_name, points_count, client)
    source_urls = extract_source(records)
    source_urls = list(set(source_urls))
    df = pd.DataFrame({"Doc_Urls": source_urls, "Exists": True})
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process collection and output path.")
    parser.add_argument("--collection_name", required=True, help="The name of the collection to process")
    parser.add_argument("--output_path", required=True, help="The path to save the output CSV")
    args = parser.parse_args()
    main(args.collection_name, args.output_path)

# run the script as shown below
#python source_search.py --collection_name 'Blogs_OpenAI_Dense' --output_path 'Blogs_Source.csv'