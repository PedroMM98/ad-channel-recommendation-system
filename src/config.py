"""Configuración simple del proyecto."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "social_media_ads_filtered.csv"
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

COLUMNAS_BASE = {
    "campaign_id": "Campaign_ID",
    "target_audience": "Target_Audience",
    "campaign_goal": "Campaign_Goal",
    "channel": "Channel_Used",
    "conversion_rate": "Conversion_Rate",
    "acquisition_cost": "Acquisition_Cost",
    "roi": "ROI",
    "clicks": "Clicks",
    "impressions": "Impressions",
    "engagement_score": "Engagement_Score",
    "date": "Date",
}
