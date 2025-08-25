import os
import json
import logging
import sqlite3
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

import requests
import pandas as pd


class FREDDataPipeline:
    """
    Pipeline for automatically pulling US debt-to-GDP data from FRED API.
    Produces a CSV with:
      - ASTDSL (short-term debt, $B)
      - ASTLL (long-term debt, $B)
      - GDP ($B)
      - ASTDSL_GDP_PCT (short-term debt % of GDP)
      - ASTLL_GDP_PCT (long-term debt % of GDP)
      - COMBINED_DEBT_GDP_PCT (sum of the two)
    """

    def __init__(self, api_key: str, db_path: str = "economic_data.db"):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.db_path = db_path
        self.setup_logging()
        self.setup_database()

        # FRED series IDs
        self.series_ids = {
            "short_term_debt": "ASTDSL",      # Federal Government; Short-Term Debt Securities ($B)
            "long_term_debt": "ASTLL",        # Federal Government; Long-Term Debt Securities ($B)
            "gdp": "GDP",                     # Gross Domestic Product ($B, SAAR quarterly)
            "total_debt_gdp": "GFDEGDQ188S",  # Total Public Debt as % of GDP (reference)
        }

    # ----------------------------
    # Setup & Storage
    # ----------------------------
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("debt_pipeline.log"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS economic_data (
                    series_id TEXT,
                    date TEXT,
                    value REAL,
                    last_updated TEXT,
                    PRIMARY KEY (series_id, date)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    series_updated TEXT,
                    status TEXT,
                    error_message TEXT
                )
                """
            )

    # ----------------------------
    # FRED helpers
    # ----------------------------
    def get_fred_data(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date

        try:
            url = f"{self.base_url}/series/observations"
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            observations = data.get("observations", [])
            if not observations:
                self.logger.warning(f"No data returned for series {series_id}")
                return None

            df = pd.DataFrame(observations)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])
            self.logger.info(f"Fetched {len(df)} observations for {series_id}")
            return df[["date", "value"]]
        except Exception as e:
            self.logger.error(f"API request failed for {series_id}: {e}")
            return None

    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """Get metadata about a FRED series."""
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        try:
            url = f"{self.base_url}/series"
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            series_info = data.get("seriess", [{}])[0]
            return series_info
        except Exception as e:
            self.logger.error(f"Error getting series info for {series_id}: {e}")
            return {}

    # ----------------------------
    # DB helpers
    # ----------------------------
    def store_data(self, series_id: str, df: pd.DataFrame) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM economic_data WHERE series_id = ?", (series_id,))
                rows = [
                    (
                        series_id,
                        row["date"].strftime("%Y-%m-%d"),
                        float(row["value"]),
                        datetime.now().isoformat(),
                    )
                    for _, row in df.iterrows()
                ]
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO economic_data (series_id, date, value, last_updated)
                    VALUES (?, ?, ?, ?)
                    """,
                    rows,
                )
                self.logger.info(f"Stored {len(rows)} rows for {series_id}")
        except Exception as e:
            self.logger.error(f"Error storing data for {series_id}: {e}")
            raise

    def get_latest_data(self, series_id: Optional[str] = None) -> pd.DataFrame:
        try:
            with sqlite3.connect(self.db_path) as conn:
                if series_id:
                    q = """
                        SELECT series_id, date, value, last_updated
                        FROM economic_data
                        WHERE series_id = ?
                        ORDER BY date ASC
                    """
                    df = pd.read_sql_query(q, conn, params=(series_id,))
                else:
                    q = """
                        SELECT series_id, date, value, last_updated
                        FROM economic_data
                        ORDER BY series_id, date ASC
                    """
                    df = pd.read_sql_query(q, conn)
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception as e:
            self.logger.error(f"Error retrieving data: {e}")
            return pd.DataFrame()

    # ----------------------------
    # Calculations & Export
    # ----------------------------
    def calculate_debt_gdp_ratios(self) -> pd.DataFrame:
        try:
            short_df = self.get_latest_data("ASTDSL")
            long_df = self.get_latest_data("ASTLL")
            gdp_df = self.get_latest_data("GDP")

            if short_df.empty or long_df.empty or gdp_df.empty:
                self.logger.error("Missing data for ratio calculations")
                return pd.DataFrame()

            short_df = short_df[["date", "value"]].rename(columns={"value": "short_term_debt"})
            long_df = long_df[["date", "value"]].rename(columns={"value": "long_term_debt"})
            gdp_df = gdp_df[["date", "value"]].rename(columns={"value": "gdp"})

            merged = short_df.merge(long_df, on="date", how="inner").merge(gdp_df, on="date", how="inner")

            merged["short_term_debt_gdp_pct"] = (merged["short_term_debt"] / merged["gdp"]) * 100.0
            merged["long_term_debt_gdp_pct"] = (merged["long_term_debt"] / merged["gdp"]) * 100.0
            merged["total_debt_gdp_pct"] = merged["short_term_debt_gdp_pct"] + merged["long_term_debt_gdp_pct"]

            merged = merged.sort_values("date").reset_index(drop=True)
            self.logger.info(f"Calculated ratios for {len(merged)} dates")
            return merged
        except Exception as e:
            self.logger.error(f"Error calculating ratios: {e}")
            return pd.DataFrame()

    def store_calculated_ratios(self, ratios_df: pd.DataFrame) -> None:
        if ratios_df.empty:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                mapping = {
                    "ASTDSL_GDP_PCT": "short_term_debt_gdp_pct",
                    "ASTLL_GDP_PCT": "long_term_debt_gdp_pct",
                    "COMBINED_DEBT_GDP_PCT": "total_debt_gdp_pct",
                }
                for series_id, col in mapping.items():
                    conn.execute("DELETE FROM economic_data WHERE series_id = ?", (series_id,))
                    rows = [
                        (
                            series_id,
                            row["date"].strftime("%Y-%m-%d"),
                            float(row[col]),
                            datetime.now().isoformat(),
                        )
                        for _, row in ratios_df.iterrows()
                    ]
                    conn.executemany(
                        """
                        INSERT INTO economic_data (series_id, date, value, last_updated)
                        VALUES (?, ?, ?, ?)
                        """,
                        rows,
                    )
            self.logger.info("Stored calculated ratios.")
        except Exception as e:
            self.logger.error(f"Error storing calculated ratios: {e}")
            raise

    def export_to_csv(self, output_path: str, include_ratios: bool = True) -> bool:
        """Export raw series and (optionally) ratio columns to CSV."""
        try:
            df = self.get_latest_data()
            if df.empty:
                self.logger.warning("No data to export")
                return False

            pivot = df.pivot(index="date", columns="series_id", values="value")

            if include_ratios:
                have_cols = all(c in pivot.columns for c in ["ASTDSL_GDP_PCT", "ASTLL_GDP_PCT", "COMBINED_DEBT_GDP_PCT"])
                if not have_cols and all(c in pivot.columns for c in ["ASTDSL", "ASTLL", "GDP"]):
                    pivot["ASTDSL_GDP_PCT"] = (pivot["ASTDSL"] / pivot["GDP"]) * 100.0
                    pivot["ASTLL_GDP_PCT"] = (pivot["ASTLL"] / pivot["GDP"]) * 100.0
                    pivot["COMBINED_DEBT_GDP_PCT"] = pivot["ASTDSL_GDP_PCT"] + pivot["ASTLL_GDP_PCT"]

            pivot = pivot.sort_index()
            pivot.to_csv(output_path)
            self.logger.info(f"Exported CSV to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting CSV: {e}")
            return False

    # ----------------------------
    # Orchestration & Reporting
    # ----------------------------
    def log_pipeline_run(self, run_id: str, updated_series: List[str], results: Dict[str, bool]) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                status = (
                    "SUCCESS"
                    if all(results.values())
                    else "PARTIAL_SUCCESS"
                    if any(results.values())
                    else "FAILED"
                )
                conn.execute(
                    """
                    INSERT INTO pipeline_runs (run_id, timestamp, series_updated, status, error_message)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (run_id, datetime.now().isoformat(), ",".join(updated_series), status, None),
                )
        except Exception as e:
            self.logger.error(f"Error logging pipeline run: {e}")

    def run_pipeline(self, lookback_years: int = 20) -> Dict[str, bool]:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")

        results: Dict[str, bool] = {}
        updated_series: List[str] = []

        self.logger.info(f"Starting run {run_id}; fetching from {start_date} to present")

        for name, sid in self.series_ids.items():
            try:
                info = self.get_series_info(sid)
                self.logger.info(f"Series {sid}: {info.get('title', 'Unknown title')}")

                df = self.get_fred_data(sid, start_date=start_date)
                if df is not None and not df.empty:
                    self.store_data(sid, df)
                    results[sid] = True
                    updated_series.append(sid)

                    latest_date = df["date"].max()
                    latest_val = float(df.loc[df["date"] == latest_date, "value"].iloc[0])
                    pretty = f"${latest_val:.1f}B" if sid in ("GDP", "ASTDSL", "ASTLL") else f"{latest_val:.2f}%"
                    self.logger.info(f"Latest {sid} = {pretty} (as of {latest_date.date()})")
                else:
                    results[sid] = False
                    self.logger.warning(f"No data for {sid}")
            except Exception as e:
                results[sid] = False
                self.logger.error(f"Failed processing {sid}: {e}")

        self.log_pipeline_run(run_id, updated_series, results)

        # Compute & persist ratios if the 3 base series succeeded
        needed = ["ASTDSL", "ASTLL", "GDP"]
        if all(results.get(s, False) for s in needed):
            self.logger.info("Calculating debt-to-GDP ratios…")
            ratios = self.calculate_debt_gdp_ratios()
            if not ratios.empty:
                self.store_calculated_ratios(ratios)
                latest = ratios.iloc[-1]
                self.logger.info(
                    "Latest ratios — Short: %.2f%%, Long: %.2f%%, Combined: %.2f%%",
                    latest["short_term_debt_gdp_pct"],
                    latest["long_term_debt_gdp_pct"],
                    latest["total_debt_gdp_pct"],
                )
        else:
            self.logger.warning("Missing required series; skipping ratio calculation")

        return results

    def get_debt_analysis_summary(self) -> Dict[str, Any]:
        try:
            ratios = self.calculate_debt_gdp_ratios()
            if ratios.empty:
                return {}

            latest = ratios.iloc[-1]
            summary = {
                "latest_date": latest["date"].strftime("%Y-%m-%d"),
                "short_term_debt_gdp": round(latest["short_term_debt_gdp_pct"], 2),
                "long_term_debt_gdp": round(latest["long_term_debt_gdp_pct"], 2),
                "combined_debt_gdp": round(latest["total_debt_gdp_pct"], 2),
                "short_term_debt_billions": round(latest["short_term_debt"], 1),
                "long_term_debt_billions": round(latest["long_term_debt"], 1),
                "gdp_billions": round(latest["gdp"], 1),
            }

            one_year_ago = latest["date"] - pd.DateOffset(years=1)
            past = ratios[ratios["date"] <= one_year_ago]
            if not past.empty:
                hist = past.iloc[-1]
                summary["yoy_change_short_term"] = round(
                    latest["short_term_debt_gdp_pct"] - hist["short_term_debt_gdp_pct"], 2
                )
                summary["yoy_change_long_term"] = round(
                    latest["long_term_debt_gdp_pct"] - hist["long_term_debt_gdp_pct"], 2
                )
                summary["yoy_change_combined"] = round(
                    latest["total_debt_gdp_pct"] - hist["total_debt_gdp_pct"], 2
                )
            return summary
        except Exception as e:
            self.logger.error(f"Error building summary: {e}")
            return {}


def main():
    # Read the key from env (works with GitHub Actions)
    API_KEY = os.getenv("FRED_API_KEY", "").strip()
    if not API_KEY:
        print("Please set FRED_API_KEY in your environment.")
        print("Get a free API key: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    pipeline = FREDDataPipeline(API_KEY)

    print("Starting debt-to-GDP data pipeline…")
    pipeline.run_pipeline(lookback_years=30)

    # Show a small summary in logs/stdout
    summary = pipeline.get_debt_analysis_summary()
    if summary:
        print(f"As of {summary['latest_date']}:")
        print(f"  Short-term Debt: ${summary['short_term_debt_billions']}B ({summary['short_term_debt_gdp']}% of GDP)")
        print(f"  Long-term Debt:  ${summary['long_term_debt_billions']}B ({summary['long_term_debt_gdp']}% of GDP)")
        print(f"  Combined Debt:   {summary['combined_debt_gdp']}% of GDP")
        print(f"  GDP:             ${summary['gdp_billions']}B")

    # Export data + ratios
    pipeline.export_to_csv("us_debt_components_analysis.csv", include_ratios=True)
    print("Exported us_debt_components_analysis.csv")


if __name__ == "__main__":
    main()
