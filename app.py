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
MAX_CONCURRENT_REQUESTS = 3  # keep concurrency low to avoid PSI quotas / timeouts

API_KEY_DEFAULT = None
try:
    API_KEY_DEFAULT = st.secrets["PAGESPEED_API_KEY"]
except Exception:
    pass

HTTP_TIMEOUT = aiohttp.ClientTimeout(total=90, connect=20, sock_read=70)

# -------------------------------------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------------------------------------

def _fmt_ms(value: Any) -> str:
    try:
        return f"{round(float(value) / 1000, 1)} s"
    except Exception:
        return "N/A"

# -------------------------------------------------------------------------------------------------
# CORE
# -------------------------------------------------------------------------------------------------
class PageSpeedAnalyzer:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required for PageSpeed Insights calls.")
        self.api_key = api_key
        self.sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def _call_api(self, session: aiohttp.ClientSession, url: str, strategy: str) -> Dict[str, Any]:
        params = {"url": url, "key": self.api_key, "strategy": strategy}
        try:
            async with self.sem:
                async with session.get(API_ENDPOINT, params=params, timeout=HTTP_TIMEOUT) as resp:
                    if resp.status != 200:
                        return {"url": url, "strategy": strategy, "success": False, "error": f"HTTP {resp.status}"}
                    payload = await resp.json()
        except asyncio.TimeoutError:
            return {"url": url, "strategy": strategy, "success": False, "error": "Timeout"}
        except Exception as e:
            return {"url": url, "strategy": strategy, "success": False, "error": str(e)}

        if "error" in payload:
            return {"url": url, "strategy": strategy, "success": False, "error": payload["error"].get("message", "API error")}

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
        async with aiohttp.ClientSession(timeout=HTTP_TIMEOUT) as session:
            tasks = [self._call_api(session, u, s) for u in urls for s in ("mobile", "desktop")]
            results: List[Dict[str, Any]] = []
            for coro in asyncio.as_completed(tasks):
                results.append(await coro)
        df = pd.DataFrame(results)
        if "success" not in df.columns:
            df["success"] = False
        return df

# -------------------------------------------------------------------------------------------------
# UI
# -------------------------------------------------------------------------------------------------

def _sidebar():
    sb = st.sidebar
    sb.header("Setup")
    sb.markdown(
        """
        **How to get a PageSpeed Insights API key**
        1. Open [Google Cloud Console](https://console.cloud.google.com/).
        2. Create or select a project.
        3. Go to **APIs & Services → Library** and enable **PageSpeed Insights API**.
        4. Navigate to **APIs & Services → Credentials** → **Create credentials ➜ API key**.
        5. Copy the generated key and paste it in the *Google API Key* field on the main screen.
        
        ---
        *Credit: **Adnan Akram***
        """
    )


def run_ui():
    st.set_page_config("PageSpeed Insights Analyzer", "🚀", "wide")
    _sidebar()

    st.title("PageSpeed Insights Analyzer")

    key_input = st.text_input("Google API Key", value=API_KEY_DEFAULT or "", type="password")
    urls_raw = st.text_area("Enter URLs (one per line)", height=150)

    if st.button("Analyze URLs", type="primary"):
        urls = [u.strip() for u in urls_raw.split("\n") if u.strip()]
        urls = [u if u.startswith("http") else f"https://{u}" for u in urls]
        if not urls:
            st.error("Please enter at least one valid URL.")
            return
        if not key_input:
            st.error("API key missing – paste yours above.")
            return

        analyzer = PageSpeedAnalyzer(key_input)
        with st.spinner("Running PageSpeed tests – may take ~1 min per URL…"):
            df = asyncio.run(analyzer.run_batch(urls))

        ok_df = df[df.success]
        if ok_df.empty:
            st.error("All requests failed.")
            st.dataframe(df)
            return

        mobile_df = ok_df[ok_df.strategy == "mobile"].reset_index(drop=True)
        desktop_df = ok_df[ok_df.strategy == "desktop"].reset_index(drop=True)

        tab_m, tab_d, tab_cmp = st.tabs(["Mobile", "Desktop", "Comparison"])

        with tab_m:
            st.dataframe(mobile_df.drop(["strategy", "success"], axis=1))
            if not mobile_df.empty:
                st.plotly_chart(px.bar(mobile_df, x="url", y="score", title="Mobile PSI Scores", color="score", range_color=[0, 100]), use_container_width=True)

        with tab_d:
            st.dataframe(desktop_df.drop(["strategy", "success"], axis=1))
            if not desktop_df.empty:
                st.plotly_chart(px.bar(desktop_df, x="url", y="score", title="Desktop PSI Scores", color="score", range_color=[0, 100]), use_container_width=True)

        with tab_cmp:
            if not mobile_df.empty and not desktop_df.empty:
                cmp = pd.DataFrame({"URL": mobile_df.url, "Mobile": mobile_df.score, "Desktop": desktop_df.score})
                cmp["Diff (Desktop–Mobile)"] = cmp["Desktop"] - cmp["Mobile"]
                st.dataframe(cmp)
                st.plotly_chart(px.bar(cmp, x="URL", y=["Mobile", "Desktop"], barmode="group", title="Mobile vs Desktop"), use_container_width=True)

        st.download_button("Download CSV", data=ok_df.to_csv(index=False), file_name="pagespeed_results.csv", mime="text/csv")


if __name__ == "__main__":
    run_ui()
