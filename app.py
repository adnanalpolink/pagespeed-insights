import streamlit as st
import pandas as pd
import asyncio
import aiohttp
import plotly.express as px
from typing import List, Dict, Any, Tuple

# -------------------------------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------------------------------
API_ENDPOINT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
MAX_CONCURRENT_REQUESTS = 5  # stay well below Google PSI default qps limits

# Try to grab key from Streamlit secrets as a sensible default
API_KEY_DEFAULT = None
try:
    API_KEY_DEFAULT = st.secrets["PAGESPEED_API_KEY"]
except Exception:
    pass

# -------------------------------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------------------------------

def _fmt_ms(value: Any) -> str:
    """Convert Lighthouse numericValue (ms) to human‑readable seconds with 1‑decimal‑place."""
    try:
        return f"{round(float(value) / 1000, 1)} s"
    except Exception:
        return "N/A"

# -------------------------------------------------------------------------------------------------
# CORE CLASS
# -------------------------------------------------------------------------------------------------
class PageSpeedAnalyzer:
    """Asynchronous PageSpeed Insights fetcher with simple concurrency guard."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required for PageSpeed Insights calls.")
        self.api_key = api_key
        self.sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def _call_api(self, session: aiohttp.ClientSession, url: str, strategy: str) -> Dict[str, Any]:
        params = {
            "url": url,
            "key": self.api_key,
            "strategy": strategy,
        }
        async with self.sem:
            async with session.get(API_ENDPOINT, params=params, timeout=45) as resp:
                # non‑200 → fast‑fail with reason text
                if resp.status != 200:
                    return {
                        "url": url,
                        "strategy": strategy,
                        "success": False,
                        "error": f"HTTP {resp.status}",
                    }
                payload = await resp.json()

        # API‑level error inside JSON
        if "error" in payload:
            err = payload["error"].get("message", "Unknown error")
            return {
                "url": url,
                "strategy": strategy,
                "success": False,
                "error": err,
            }

        lh = payload.get("lighthouseResult", {})
        perf = lh.get("categories", {}).get("performance", {})
        audits = lh.get("audits", {})

        return {
            "url": url,
            "strategy": strategy,
            "success": True,
            "score": round((perf.get("score", 0) or 0) * 100, 1),
            "firstContentfulPaint": _fmt_ms(audits.get("first-contentful-paint", {}).get("numericValue")),
            "speedIndex": _fmt_ms(audits.get("speed-index", {}).get("numericValue")),
            "timeToInteractive": _fmt_ms(audits.get("interactive", {}).get("numericValue")),
            "largestContentfulPaint": _fmt_ms(audits.get("largest-contentful-paint", {}).get("numericValue")),
            "totalBlockingTime": _fmt_ms(audits.get("total-blocking-time", {}).get("numericValue")),
            "cumulativeLayoutShift": audits.get("cumulative-layout-shift", {}).get("displayValue", "N/A"),
        }

    async def run_batch(self, urls: List[str]) -> pd.DataFrame:
        async with aiohttp.ClientSession() as session:
            tasks = [self._call_api(session, u, s) for u in urls for s in ("mobile", "desktop")]
            results = await asyncio.gather(*tasks)
        df = pd.DataFrame(results)
        if "success" not in df.columns:
            df["success"] = False
        return df

# -------------------------------------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------------------------------------

def run_ui():
    st.set_page_config("PageSpeed Insights Analyzer", "🚀", "wide")
    st.title("PageSpeed Insights Analyzer")
    st.write("Analyze performance metrics for multiple URLs using the Google PageSpeed Insights API.")

    key_input = st.text_input("Google API Key", value=API_KEY_DEFAULT or "", type="password")
    urls_raw = st.text_area("Enter URLs (one per line)", height=150)

    if st.button("Analyze URLs", type="primary"):
        urls = [u.strip() for u in urls_raw.split("\n") if u.strip()]
        urls = [u if u.startswith("http") else f"https://{u}" for u in urls]
        if not urls:
            st.error("Please enter at least one valid URL.")
            return
        if not key_input:
            st.error("An API key is required. Paste yours above – get it from Google Cloud Console → PageSpeed Insights API.")
            return

        analyzer = PageSpeedAnalyzer(key_input)
        with st.spinner("Running PageSpeed tests… this can take a minute"):
            df = asyncio.run(analyzer.run_batch(urls))

        # Split & visualise
        success_df = df[df.success]
        if success_df.empty:
            st.error("All requests failed. Check your URLs and API key.")
            st.dataframe(df)
            return

        mobile_df = success_df[success_df.strategy == "mobile"].reset_index(drop=True)
        desktop_df = success_df[success_df.strategy == "desktop"].reset_index(drop=True)

        tab_m, tab_d, tab_cmp = st.tabs(["Mobile", "Desktop", "Comparison"])

        with tab_m:
            st.header("Mobile results")
            st.dataframe(mobile_df.drop(["strategy", "success"], axis=1))
            if not mobile_df.empty:
                st.plotly_chart(px.bar(mobile_df, x="url", y="score", title="Mobile PSI Scores", color="score", range_color=[0, 100]), use_container_width=True)

        with tab_d:
            st.header("Desktop results")
            st.dataframe(desktop_df.drop(["strategy", "success"], axis=1))
            if not desktop_df.empty:
                st.plotly_chart(px.bar(desktop_df, x="url", y="score", title="Desktop PSI Scores", color="score", range_color=[0, 100]), use_container_width=True)

        with tab_cmp:
            if not mobile_df.empty and not desktop_df.empty:
                cmp = pd.DataFrame({
                    "URL": mobile_df.url,
                    "Mobile": mobile_df.score,
                    "Desktop": desktop_df.score,
                })
                cmp["Diff (Desktop–Mobile)"] = cmp["Desktop"] - cmp["Mobile"]
                st.dataframe(cmp)
                st.plotly_chart(px.bar(cmp, x="URL", y=["Mobile", "Desktop"], barmode="group", title="Mobile vs Desktop"), use_container_width=True)

        st.download_button("Download CSV", data=success_df.to_csv(index=False), file_name="pagespeed_results.csv", mime="text/csv")


if __name__ == "__main__":
    run_ui()
