import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import sqlite3
import os
import logging
from typing import Optional, Dict, Any

class FREDDataPipeline:
    """
    Pipeline for automatically pulling US debt-to-GDP data from FRED API
    """
    
    def __init__(self, api_key: str, db_path: str = "economic_data.db"):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.db_path = db_path
        self.setup_logging()
        self.setup_database()
        
        # Key FRED series IDs for debt-to-GDP analysis
        self.series_ids = {
            'short_term_debt': 'ASTDSL',      # Federal Government; Short-Term Debt Securities (billions)
            'long_term_debt': 'ASTLL',        # Federal Government; Long-Term Debt Securities (billions)  
            'gdp': 'GDP',                     # Gross Domestic Product (billions)
            'total_debt_gdp': 'GFDEGDQ188S',  # Federal Debt: Total Public Debt as Percent of GDP (for reference)
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('debt_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Initialize SQLite database for storing data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS economic_data (
                    series_id TEXT,
                    date TEXT,
                    value REAL,
                    last_updated TEXT,
                    PRIMARY KEY (series_id, date)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    series_updated TEXT,
                    status TEXT,
                    error_message TEXT
                )
            ''')
    
    def get_fred_data(self, series_id: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch data from FRED API for a specific series
        
        Args:
            series_id: FRED series identifier
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
        
        Returns:
            DataFrame with date and value columns, or None if error
        """
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
            
        try:
            url = f"{self.base_url}/series/observations"
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            observations = data.get('observations', [])
            
            if not observations:
                self.logger.warning(f"No data returned for series {series_id}")
                return None
            
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])  # Remove missing values
            
            self.logger.info(f"Successfully fetched {len(df)} observations for {series_id}")
            return df[['date', 'value']]
            
        except requests.RequestException as e:
            self.logger.error(f"API request failed for {series_id}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing data for {series_id}: {str(e)}")
            return None
    
    def store_data(self, series_id: str, df: pd.DataFrame):
        """Store data in SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Delete existing data for this series to avoid duplicates
                conn.execute('DELETE FROM economic_data WHERE series_id = ?', (series_id,))
                
                # Insert new data
                for _, row in df.iterrows():
                    conn.execute('''
                        INSERT OR REPLACE INTO economic_data 
                        (series_id, date, value, last_updated) 
                        VALUES (?, ?, ?, ?)
                    ''', (series_id, row['date'].strftime('%Y-%m-%d'), 
                          row['value'], datetime.now().isoformat()))
                
                self.logger.info(f"Stored {len(df)} records for {series_id}")
                
        except Exception as e:
            self.logger.error(f"Error storing data for {series_id}: {str(e)}")
            raise
    
    def calculate_debt_gdp_ratios(self) -> pd.DataFrame:
        """
        Calculate debt-to-GDP ratios for short-term, long-term, and combined debt
        
        Returns:
            DataFrame with calculated ratios as percentages
        """
        try:
            # Get data for all series
            short_term_df = self.get_latest_data('ASTDSL')
            long_term_df = self.get_latest_data('ASTLL') 
            gdp_df = self.get_latest_data('GDP')
            
            if short_term_df.empty or long_term_df.empty or gdp_df.empty:
                self.logger.error("Missing data for ratio calculations")
                return pd.DataFrame()
            
            # Prepare dataframes
            short_term_df = short_term_df[['date', 'value']].rename(columns={'value': 'short_term_debt'})
            long_term_df = long_term_df[['date', 'value']].rename(columns={'value': 'long_term_debt'})
            gdp_df = gdp_df[['date', 'value']].rename(columns={'value': 'gdp'})
            
            # Merge on date (inner join to only keep dates where all data exists)
            merged_df = short_term_df.merge(long_term_df, on='date', how='inner')
            merged_df = merged_df.merge(gdp_df, on='date', how='inner')
            
            # Calculate ratios as percentages
            merged_df['short_term_debt_gdp_pct'] = (merged_df['short_term_debt'] / merged_df['gdp']) * 100
            merged_df['long_term_debt_gdp_pct'] = (merged_df['long_term_debt'] / merged_df['gdp']) * 100
            merged_df['total_debt_gdp_pct'] = merged_df['short_term_debt_gdp_pct'] + merged_df['long_term_debt_gdp_pct']
            
            # Sort by date
            merged_df = merged_df.sort_values('date')
            
            self.logger.info(f"Calculated debt-to-GDP ratios for {len(merged_df)} periods")
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error calculating debt-to-GDP ratios: {str(e)}")
            return pd.DataFrame()
    
    def store_calculated_ratios(self, ratios_df: pd.DataFrame):
        """Store calculated ratios in the database"""
        if ratios_df.empty:
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store calculated ratios as separate series
                ratio_series = {
                    'ASTDSL_GDP_PCT': 'short_term_debt_gdp_pct',
                    'ASTLL_GDP_PCT': 'long_term_debt_gdp_pct', 
                    'COMBINED_DEBT_GDP_PCT': 'total_debt_gdp_pct'
                }
                
                for series_id, column_name in ratio_series.items():
                    # Delete existing calculated data
                    conn.execute('DELETE FROM economic_data WHERE series_id = ?', (series_id,))
                    
                    # Insert new calculated data
                    for _, row in ratios_df.iterrows():
                        conn.execute('''
                            INSERT INTO economic_data 
                            (series_id, date, value, last_updated) 
                            VALUES (?, ?, ?, ?)
                        ''', (series_id, row['date'].strftime('%Y-%m-%d'), 
                              row[column_name], datetime.now().isoformat()))
                
                self.logger.info(f"Stored calculated ratios for {len(ratios_df)} periods")
                
        except Exception as e:
            self.logger.error(f"Error storing calculated ratios: {str(e)}")
            raise
        """Get metadata about a FRED series"""
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        
        try:
            url = f"{self.base_url}/series"
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            series_info = data.get('seriess', [{}])[0]
            return series_info
            
        except Exception as e:
            self.logger.error(f"Error getting series info for {series_id}: {str(e)}")
            return {}
    
    def run_pipeline(self, lookback_years: int = 20) -> Dict[str, bool]:
        """
        Execute the full data pipeline
        
        Args:
            lookback_years: How many years of historical data to fetch
            
        Returns:
            Dictionary with success status for each series
        """
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime('%Y-%m-%d')
        
        results = {}
        updated_series = []
        
        self.logger.info(f"Starting pipeline run {run_id}")
        self.logger.info(f"Fetching data from {start_date} to present")
        
        for series_name, series_id in self.series_ids.items():
            try:
                self.logger.info(f"Processing {series_name} ({series_id})")
                
                # Get series metadata
                series_info = self.get_series_info(series_id)
                self.logger.info(f"Series: {series_info.get('title', 'Unknown')}")
                
                # Fetch data
                df = self.get_fred_data(series_id, start_date=start_date)
                
                if df is not None and not df.empty:
                    self.store_data(series_id, df)
                    results[series_id] = True
                    updated_series.append(series_id)
                    
                    # Log summary statistics
                    latest_date = df['date'].max()
                    latest_value = df[df['date'] == latest_date]['value'].iloc[0]
                    
                    if series_id == 'GDP':
                        self.logger.info(f"Latest {series_name}: ${latest_value:.1f} billion (as of {latest_date.strftime('%Y-%m-%d')})")
                    elif series_id in ['ASTDSL', 'ASTLL']:
                        self.logger.info(f"Latest {series_name}: ${latest_value:.1f} billion (as of {latest_date.strftime('%Y-%m-%d')})")
                    else:
                        self.logger.info(f"Latest {series_name}: {latest_value}% (as of {latest_date.strftime('%Y-%m-%d')})")
                    
                else:
                    results[series_id] = False
                    self.logger.warning(f"No data retrieved for {series_id}")
                    
            except Exception as e:
                results[series_id] = False
                self.logger.error(f"Failed to process {series_id}: {str(e)}")
        
        # Log pipeline run
        self.log_pipeline_run(run_id, updated_series, results)
        
        # Calculate and store debt-to-GDP ratios if we have the required data
        if all(results.get(series_id, False) for series_id in ['ASTDSL', 'ASTLL', 'GDP']):
            self.logger.info("Calculating debt-to-GDP ratios...")
            ratios_df = self.calculate_debt_gdp_ratios()
            if not ratios_df.empty:
                self.store_calculated_ratios(ratios_df)
                
                # Log latest calculated ratios
                latest_ratios = ratios_df.iloc[-1]
                self.logger.info(f"Latest Short-term Debt/GDP: {latest_ratios['short_term_debt_gdp_pct']:.2f}%")
                self.logger.info(f"Latest Long-term Debt/GDP: {latest_ratios['long_term_debt_gdp_pct']:.2f}%") 
                self.logger.info(f"Latest Combined Debt/GDP: {latest_ratios['total_debt_gdp_pct']:.2f}%")
        else:
            self.logger.warning("Cannot calculate ratios - missing required data series")
        
        return results
    
    def log_pipeline_run(self, run_id: str, updated_series: list, results: Dict[str, bool]):
        """Log the pipeline run results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                status = "SUCCESS" if all(results.values()) else "PARTIAL_SUCCESS" if any(results.values()) else "FAILED"
                
                conn.execute('''
                    INSERT INTO pipeline_runs 
                    (run_id, timestamp, series_updated, status, error_message) 
                    VALUES (?, ?, ?, ?, ?)
                ''', (run_id, datetime.now().isoformat(), 
                      ','.join(updated_series), status, None))
                
        except Exception as e:
            self.logger.error(f"Error logging pipeline run: {str(e)}")
    
    def get_latest_data(self, series_id: str = None) -> pd.DataFrame:
        """Retrieve latest data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if series_id:
                    query = '''
                        SELECT series_id, date, value, last_updated 
                        FROM economic_data 
                        WHERE series_id = ? 
                        ORDER BY date DESC
                    '''
                    df = pd.read_sql_query(query, conn, params=(series_id,))
                else:
                    query = '''
                        SELECT series_id, date, value, last_updated 
                        FROM economic_data 
                        ORDER BY series_id, date DESC
                    '''
                    df = pd.read_sql_query(query, conn)
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                
                return df
                
        except Exception as e:
            self.logger.error(f"Error retrieving data: {str(e)}")
            return pd.DataFrame()
    
    def get_debt_analysis_summary(self) -> Dict[str, Any]:
        """Get summary analysis of debt components"""
        try:
            ratios_df = self.calculate_debt_gdp_ratios()
            if ratios_df.empty:
                return {}
            
            latest_data = ratios_df.iloc[-1]
            
            # Calculate trends (compare to 1 year ago if available)
            one_year_ago = latest_data['date'] - pd.DateOffset(years=1)
            historical_data = ratios_df[ratios_df['date'] <= one_year_ago]
            
            summary = {
                'latest_date': latest_data['date'].strftime('%Y-%m-%d'),
                'short_term_debt_gdp': round(latest_data['short_term_debt_gdp_pct'], 2),
                'long_term_debt_gdp': round(latest_data['long_term_debt_gdp_pct'], 2),
                'combined_debt_gdp': round(latest_data['total_debt_gdp_pct'], 2),
                'short_term_debt_billions': round(latest_data['short_term_debt'], 1),
                'long_term_debt_billions': round(latest_data['long_term_debt'], 1),
                'gdp_billions': round(latest_data['gdp'], 1)
            }
            
            # Add year-over-year changes if data available
            if not historical_data.empty:
                historical_latest = historical_data.iloc[-1]
                summary['yoy_change_short_term'] = round(
                    latest_data['short_term_debt_gdp_pct'] - historical_latest['short_term_debt_gdp_pct'], 2)
                summary['yoy_change_long_term'] = round(
                    latest_data['long_term_debt_gdp_pct'] - historical_latest['long_term_debt_gdp_pct'], 2)
                summary['yoy_change_combined'] = round(
                    latest_data['total_debt_gdp_pct'] - historical_latest['total_debt_gdp_pct'], 2)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating debt analysis summary: {str(e)}")
            return {}
        """Export all data to CSV file"""
        try:
            df = self.get_latest_data()
            if not df.empty:
                # Pivot to have series as columns
                pivot_df = df.pivot(index='date', columns='series_id', values='value')
                pivot_df.to_csv(output_path)
                self.logger.info(f"Data exported to {output_path}")
                return True
            else:
                self.logger.warning("No data available for export")
                return False
                
        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}")
            return False


def main():
    """
    Example usage of the debt-to-GDP pipeline
    """
    # You need to get a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
    API_KEY = "YOUR_FRED_API_KEY_HERE"
    
    if API_KEY == "YOUR_FRED_API_KEY_HERE":
        print("Please set your FRED API key in the API_KEY variable")
        print("Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return
    
    # Initialize pipeline
    pipeline = FREDDataPipeline(API_KEY)
    
    # Run the pipeline
    print("Starting debt-to-GDP data pipeline...")
    results = pipeline.run_pipeline(lookback_years=30)
    
    # Print results
    print("\nPipeline Results:")
    for series_id, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {series_id}")
    
    # Display latest data
    print("\nLatest Debt Component Analysis:")
    summary = pipeline.get_debt_analysis_summary()
    if summary:
        print(f"As of {summary['latest_date']}:")
        print(f"  Short-term Debt: ${summary['short_term_debt_billions']} billion ({summary['short_term_debt_gdp']}% of GDP)")
        print(f"  Long-term Debt:  ${summary['long_term_debt_billions']} billion ({summary['long_term_debt_gdp']}% of GDP)")
        print(f"  Combined Debt:   {summary['combined_debt_gdp']}% of GDP")
        print(f"  GDP: ${summary['gdp_billions']} billion")
        
        if 'yoy_change_combined' in summary:
            print(f"\nYear-over-Year Changes:")
            print(f"  Short-term: {summary['yoy_change_short_term']:+.2f}% of GDP")
            print(f"  Long-term:  {summary['yoy_change_long_term']:+.2f}% of GDP") 
            print(f"  Combined:   {summary['yoy_change_combined']:+.2f}% of GDP")
    
    # Compare to total debt for reference
    total_debt_data = pipeline.get_latest_data('GFDEGDQ188S')
    if not total_debt_data.empty:
        latest_total = total_debt_data.iloc[0]
        print(f"\nFor comparison:")
        print(f"  Total Public Debt (FRED): {latest_total['value']:.1f}% of GDP (as of {latest_total['date'].strftime('%Y-%m-%d')})")
    
    # Export data
    pipeline.export_to_csv("us_debt_components_analysis.csv", include_ratios=True)
    print("\nData exported to us_debt_components_analysis.csv")


if __name__ == "__main__":
    main()
