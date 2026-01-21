"""
NBA Score Prediction - Data Exploration
=========================================

This script explores the Kaggle NBA SQLite database and provides
an overview of available data for the prediction project.

Usage:
    python src/data_processing/01_explore_data.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

# Configuration
DATA_DIR = Path("data/raw")
DB_PATH = DATA_DIR / "basketball.sqlite"
OUTPUT_DIR = Path("data/processed")

def check_database_exists():
    """Check if the database file exists"""
    if not DB_PATH.exists():
        print("❌ Database not found!")
        print(f"   Expected location: {DB_PATH}")
        print("\n📥 Please download the Kaggle dataset:")
        print("   https://www.kaggle.com/datasets/wyattowalsh/basketball")
        print(f"   And place basketball.sqlite in {DATA_DIR}/")
        return False
    
    print(f"✓ Database found: {DB_PATH}")
    return True

def explore_database():
    """Explore the database structure and content"""
    
    print("\n" + "="*70)
    print("NBA DATABASE EXPLORATION")
    print("="*70)
    
    conn = sqlite3.connect(DB_PATH)
    
    # List all tables
    print("\n📊 Available Tables:")
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", 
        conn
    )
    
    for idx, table in enumerate(tables['name'], 1):
        print(f"   {idx}. {table}")
    
    # Explore each table
    print("\n" + "="*70)
    print("TABLE DETAILS")
    print("="*70)
    
    for table in tables['name']:
        print(f"\n📋 Table: {table}")
        print("-" * 70)
        
        # Get row count
        count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn)
        print(f"   Rows: {count['count'].iloc[0]:,}")
        
        # Get column info
        columns = pd.read_sql(f"PRAGMA table_info({table})", conn)
        print(f"   Columns ({len(columns)}):")
        for _, col in columns.iterrows():
            print(f"      - {col['name']} ({col['type']})")
        
        # Show sample data
        print(f"\n   Sample Data:")
        sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", conn)
        print(sample.to_string(index=False))
    
    conn.close()
    
    print("\n" + "="*70)
    print("✓ Exploration complete!")
    print("="*70)

def main():
    """Main execution"""
    print("🏀 NBA Score Prediction - Data Exploration")
    print()
    
    if not check_database_exists():
        return
    
    explore_database()
    
    print("\n📝 Next Steps:")
    print("   1. Review the table structures above")
    print("   2. Identify which tables contain game scores")
    print("   3. Run feature engineering scripts")

if __name__ == "__main__":
    main()
