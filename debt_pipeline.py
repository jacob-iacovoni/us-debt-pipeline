import os
import logging
import sqlite3
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

import requests
import pandas as pd


class FREDDataPipeline:
    """
    Pulls debt & GDP data from FRED, stores to SQLite, computes ratios, and exports CSV.

    Output CSV columns (where available):
      Levels (billions): ASTDSL, ASTLL, TCMDO, GDP
      Ratios  (% GDP) : ASTDSL_GDP_PCT, ASTLL_GDP_PCT, COMBINED_DEBT_GDP_PCT, TCMDO_GDP_PCT, PRIVATE_DEBT_GDP_PCT
    """

    def __init__(self, api_key: str, db_path: str = "economic_data.db"):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.db_path = db_path
        self.setup_logging()
        self.setup_database()

        # FRED series:
        # - ASTDSL: All Sectors; Debt Securities; Liability, Level (millions)
        # - ASTLL : All Sectors; Loans; Liability, Level (millions)
        # - TCMDO : All Sectors; Debt Securities and Loans; Liability, Level (millions)
        # - GDP   : Gross Domestic Product (billions, SAAR)
        # - QUSPAM770A: Private Non-Financial Sector Credit, % of GDP (already percent)
        self.series_ids = {
            "debt_securities_all": "ASTDSL",
            "loans_all": "ASTLL",
            "all_sectors_debt": "TCMDO",
            "gdp": "GDP",
            "private_debt_gdp_pct": "QUSPAM770A",
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
    def _scale_to_billions(self, series_id: str, df: pd.DataFrame) -> pd.DataFrame:
        """Scale Z.1 'level' series (in millions) to billions to match GDP units."""
        in_millions = {"ASTDSL", "ASTLL", "TCMDO"}
        if series_id in in_millions and not df.empty:
            df = df.copy()
            df["value"] = df["value"] / 1000.0  # millions -> billions
        return df

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
            df = df[["date", "value"]]
            df = self._scale_to_billions(series_id, df)  # normalize units
            self.logger.info(f"Fetched {len(df)} observations for {series_id}")
            return df
        except Exception as e:
            self.logger.error(f"API request failed for {series_id}: {e}")
            return None

    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        params = {"series_id": series_id, "api_key": self.api_key, "file_type": "json"}
        try:
            url = f"{self.base_url}/series"
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("seriess", [{}])[0]
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
        """
        Build panel with:
          Levels (billions): ASTDSL_B, ASTLL_B, TCMDO_B, GDP_B
          Ratios  (% GDP) : ASTDSL_GDP_PCT, ASTLL_GDP_PCT, COMBINED_DEBT_GDP_PCT, TCMDO_GDP_PCT, PRIVATE_DEBT_GDP_PCT
        """
        try:
            ast_dsec = self.get_latest_data("ASTDSL")      # billions (scaled)
            ast_loans = self.get_latest_data("ASTLL")      # billions (scaled)
            tcmdo     = self.get_latest_data("TCMDO")      # billions (scaled)
            gdp       = self.get_latest_data("GDP")        # billions
            priv_pct  = self.get_latest_data("QUSPAM770A") # percent

            if any(df.empty for df in [ast_dsec, ast_loans, gdp]):
                self.logger.error("Missing core series for ratio calculations (ASTDSL/ASTLL/GDP).")
                return pd.DataFrame()

            dsec  = ast_dsec[["date", "value"]].rename(columns={"value": "ASTDSL_B"})
            loans = ast_loans[["date", "value"]].rename(columns={"value": "ASTLL_B"})
            gdpdf = gdp[["date", "value"]].rename(columns={"value": "GDP_B"})

            merged = dsec.merge(loans, on="date", how="inner").merge(gdpdf, on="date", how="inner")

            if not tcmdo.empty:
                merged = merged.merge(
                    tcmdo[["date", "value"]].rename(columns={"value": "TCMDO_B"}), on="date", how="left"
                )

            if not priv_pct.empty:
                merged = merged.merge(
                    priv_pct[["date", "value"]].rename(columns={"value": "PRIVATE_DEBT_GDP_PCT"}),
                    on="date",
                    how="left",
                )

            # Ratios
            merged["ASTDSL_GDP_PCT"] = (merged["ASTDSL_B"] / merged["GDP_B"]) * 100.0
            merged["ASTLL_GDP_PCT"]  = (merged["ASTLL_B"]  / merged["GDP_B"]) * 100.0
            merged["COMBINED_DEBT_GDP_PCT"] = merged["ASTDSL_GDP_PCT"] + merged["ASTLL_GDP_PCT"]
            if "TCMDO_B" in merged.columns:
                merged["TCMDO_GDP_PCT"] = (merged["TCMDO_B"] / merged["GDP_B"]) * 100.0

            merged = merged.sort_values("date").reset_index(drop=True)
            self.logger.info(f"Calculated ratios for {len(merged)} quarters.")
            return merged
        except Exception as e:
            self.logger.error(f"Error calculating debt/GDP ratios: {e}")
            return pd.DataFrame()

    def store_calculated_ratios(self, ratios_df: pd.DataFrame):
        if ratios_df.empty:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                mapping = {
                    "ASTDSL_GDP_PCT": "ASTDSL_GDP_PCT",
                    "ASTLL_GDP_PCT": "ASTLL_GDP_PCT",
                    "COMBINED_DEBT_GDP_PCT": "COMBINED_DEBT_GDP_PCT",
                    "TCMDO_GDP_PCT": "TCMDO_GDP_PCT",
                    "PRIVATE_DEBT_GDP_PCT": "PRIVATE_DEBT_GDP_PCT",
                }
                for series_id, col in mapping.items():
                    if col not in ratios_df.columns:
                        continue
                    conn.execute("DELETE FROM economic_data WHERE series_id = ?", (series_id,))
                    rows = [
                        (series_id, row["date"].strftime("%Y-%m-%d"), float(row[col]), datetime.now().isoformat())
                        for _, row in ratios_df.iterrows()
                        if pd.notna(row.get(col))
                    ]
                    conn.executemany(
                        "INSERT INTO economic_data (series_id, date, value, last_updated) VALUES (?, ?, ?, ?)",
                        rows,
                    )
            self.logger.info("Stored calculated ratio series.")
        except Exception as e:
            self.logger.error(f"Error storing calculated ratios: {e}")
            raise

    def export_to_csv(self, output_path: str, include_ratios: bool = True) -> bool:
        """Export raw series and (optionally) computed ratio columns to CSV."""
        try:
            df = self.get_latest_data()
            if df.empty:
                self.logger.warning("No data to export")
                return False

            pivot = df.pivot(index="date", columns="series_id", values="value")

            # Compute ratios on the fly if not persisted
            have_base = all(c in pivot.columns for c in ["ASTDSL", "ASTLL", "GDP"])
            if include_ratios and have_base:
                if "ASTDSL_GDP_PCT" not in pivot.columns:
                    pivot["ASTDSL_GDP_PCT"] = (pivot["ASTDSL"] / pivot["GDP"]) * 100.0
                if "ASTLL_GDP_PCT" not in pivot.columns:
                    pivot["ASTLL_GDP_PCT"] = (pivot["ASTLL"] / pivot["GDP"]) * 100.0
                if "COMBINED_DEBT_GDP_PCT" not in pivot.columns:
                    pivot["COMBINED_DEBT_GDP_PCT"] = pivot["ASTDSL_GDP_PCT"] + pivot["ASTLL_GDP_PCT"]
                if "TCMDO" in pivot.columns and "TCMDO_GDP_PCT" not in pivot.columns:
                    pivot["TCMDO_GDP_PCT"] = (pivot["TCMDO"] / pivot["GDP"]) * 100.0
                # PRIVATE_DEBT_GDP_PCT is already a percent if present

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

    def run_pipeline(self, lookback_years: int = 30) -> Dict[str, bool]:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")

        results: Dict[str, bool] = {}
        updated_series: List[str] = []

        self.logger.info(f"Starting run {run_id}; fetching from {start_date} to present")

        # Fetch & store for each series id
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
                    if sid in ("ASTDSL", "ASTLL", "TCMDO", "GDP"):
                        pretty = f"${latest_val:.1f}B"
                    else:
                        pretty = f"{latest_val:.2f}%"
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
                    "Latest ratios — Securities: %.2f%%, Loans: %.2f%%, Combined: %.2f%%",
                    latest["ASTDSL_GDP_PCT"],
                    latest["ASTLL_GDP_PCT"],
                    latest["COMBINED_DEBT_GDP_PCT"],
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
                "short_term_debt_gdp": round(latest["ASTDSL_GDP_PCT"], 2),
                "long_term_debt_gdp": round(latest["ASTLL_GDP_PCT"], 2),
                "combined_debt_gdp": round(latest["COMBINED_DEBT_GDP_PCT"], 2),
                "gdp_billions": round(latest["GDP_B"], 1),
                "ASTDSL_billions": round(latest["ASTDSL_B"], 1),
                "ASTLL_billions": round(latest["ASTLL_B"], 1),
            }
            if "TCMDO_B" in latest:
                summary["TCMDO_billions"] = round(latest["TCMDO_B"], 1)
                if "TCMDO_GDP_PCT" in latest:
                    summary["TCMDO_debt_gdp"] = round(latest["TCMDO_GDP_PCT"], 2)
            if "PRIVATE_DEBT_GDP_PCT" in latest and pd.notna(latest["PRIVATE_DEBT_GDP_PCT"]):
                summary["private_debt_gdp"] = round(latest["PRIVATE_DEBT_GDP_PCT"], 2)
            return summary
        except Exception as e:
            self.logger.error(f"Error building summary: {e}")
            return {}


def main():
    # Read API key from env (works with GitHub Actions secrets)
    API_KEY = os.getenv("FRED_API_KEY", "").strip()
    if not API_KEY:
        print("Please set FRED_API_KEY in your environment.")
        print("Get a free API key: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    pipeline = FREDDataPipeline(API_KEY)

    print("Starting debt-to-GDP data pipeline…")
    pipeline.run_pipeline(lookback_years=30)

    # Show a small summary
    summary = pipeline.get_debt_analysis_summary()
    if summary:
        print(f"As of {summary['latest_date']}:")
        print(f"  Securities (ASTDSL): ${summary['ASTDSL_billions']}B ({summary['short_term_debt_gdp']}% of GDP)")
        print(f"  Loans (ASTLL):       ${summary['ASTLL_billions']}B ({summary['long_term_debt_gdp']}% of GDP)")
        print(f"  Combined (S+L):      {summary['combined_debt_gdp']}% of GDP")
        print(f"  GDP:                 ${summary['gdp_billions']}B")
        if "TCMDO_debt_gdp" in summary:
            print(f"  TCMDO:               {summary['TCMDO_debt_gdp']}% of GDP")
        if "private_debt_gdp" in summary:
            print(f"  Private debt (BIS):  {summary['private_debt_gdp']}% of GDP")

    # Export data + ratios
    pipeline.export_to_csv("us_debt_components_analysis.csv", include_ratios=True)
    print("Exported us_debt_components_analysis.csv")


if __name__ == "__main__":
    main()
