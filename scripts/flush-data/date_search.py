from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os
from datetime import datetime
import re

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


def detect_date_format(date_str):
    date_formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d", "%b %d, %Y", "%d %b %Y",
        "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%a, %d %b %Y %H:%M:%S %z"
    ]
    for fmt in date_formats:
        try:
            datetime.strptime(date_str, fmt)
            return fmt
        except ValueError:
            continue
    return None


def extract_date(records):
    publication_dates = []
    for record_group, _ in records:
        for rec in record_group:
            if 'metadata' in rec.payload:
                # Find keys containing 'date' (case-insensitive)
                date_keys = [k for k in rec.payload['metadata'].keys() if 'date' in k.lower()]
                for date_key in date_keys:
                    date_value = rec.payload['metadata'].get(date_key)
                    if date_value:
                        # Store each date with its corresponding point ID
                        publication_dates.append((rec.id, date_value))

    print([date for _, date in publication_dates])  # Print only the dates for clarity
    date_format = " "
    if publication_dates:
        date_format = detect_date_format(publication_dates[0][1])
    print(f"Detected date format: {date_format}")
    return publication_dates, date_format


def get_oldest_months(dates_with_ids, date_format, year,cut_off_months):
    # Parse dates, filter by year, and keep point IDs
    parsed_dates_with_ids = [
        (point_id, datetime.strptime(date, date_format)) for point_id, date in dates_with_ids
        if datetime.strptime(date, date_format).year == year
    ]

    # Sort dates and get unique months with point IDs
    unique_months = {}
    for point_id, date in parsed_dates_with_ids:
        year_month = (date.year, date.month)
        if year_month not in unique_months:
            unique_months[year_month] = []
        unique_months[year_month].append(point_id)

    # Get the oldest three unique months
    oldest_months = sorted(unique_months.keys())[:cut_off_months]
    points=[]
    # Display results with point IDs for each month
    for year, month in oldest_months:
        month_name = datetime(year, month, 1).strftime('%B %Y')
        point_ids = unique_months[(year, month)]
        print(f"{month_name}: Points IDs -> {point_ids}")
        points.append(point_ids)

    return points


def get_oldest_days(dates_with_ids, date_format, year, cut_off_days):
    # Parse and filter dates based on the year
    parsed_dates_with_ids = [
        (point_id, datetime.strptime(date, date_format)) for point_id, date in dates_with_ids
        if datetime.strptime(date, date_format).year == year
    ]

    # Sort by date and select the oldest dates within the cutoff days
    sorted_dates_with_ids = sorted(parsed_dates_with_ids, key=lambda x: x[1])
    oldest_days_with_ids = sorted_dates_with_ids[:cut_off_days]

    # Collect only the point IDs from the oldest days
    points = [point_id for point_id, _ in oldest_days_with_ids]

    # Print results
    for point_id, date in oldest_days_with_ids:
        print(f"{date.strftime('%Y-%m-%d')}: Point ID -> {point_id}")

    return points

def find_oldest_and_most_recent_dates(dates_with_ids, date_format):
    """
    Finds the most recent and oldest dates from a list of (point_id, date) tuples.
    """
    # Parse the dates
    parsed_dates_with_ids = [
        (point_id, datetime.strptime(date, date_format)) for point_id, date in dates_with_ids
    ]

    # Sort the dates
    sorted_dates_with_ids = sorted(parsed_dates_with_ids, key=lambda x: x[1])

    # Oldest and most recent
    oldest_date_entry = sorted_dates_with_ids[0]
    most_recent_date_entry = sorted_dates_with_ids[-1]

    # Print the results
    print(f"Oldest Date: {oldest_date_entry[1].strftime('%Y-%m-%d')} -> Point ID: {oldest_date_entry[0]}")
    print(f"Most Recent Date: {most_recent_date_entry[1].strftime('%Y-%m-%d')} -> Point ID: {most_recent_date_entry[0]}")

    return oldest_date_entry, most_recent_date_entry



def delete_entries_by_points_id(client, collection_name, points_ids):
    # Perform deletion
    client.delete(
        collection_name=collection_name,
        points_selector={"ids": points_ids}
    )
    print(f"Deleted {len(points_ids)} entries from the collection '{collection_name}'.")



if __name__=="__main__":
    client = init_qdrant_client()
    collection_name = "Blogs"
    points_count = count_data(collection_name, client)
    records = extract_payload(collection_name, points_count, client)
    dates_with_ids, date_format = extract_date(records)
    find_oldest_and_most_recent_dates(dates_with_ids,date_format)
    point_ids=get_oldest_days(dates_with_ids, date_format, 2024,40)
    # point_ids = get_oldest_months(dates_with_ids, date_format, 2024, 2)
    # delete_entries_by_points_id(client,collection_name,point_ids)
