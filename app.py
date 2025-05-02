import streamlit as st
import requests
import pandas as pd
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px
from typing import List, Dict, Any, Union, Tuple
import os

try:
    API_KEY = st.secrets["PAGESPEED_API_KEY"]
except Exception:
    API_KEY = None
# Configuration
API_ENDPOINT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
MAX_CONCURRENT_REQUESTS = 5  # Adjust based on API limits

class PageSpeedAnalyzer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEY
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
    async def check_status_code(self, session: aiohttp.ClientSession, url: str) -> Tuple[str, int]:
        """Verify URL returns valid status code before analysis"""
        try:
            async with self.semaphore:
                async with session.head(url, allow_redirects=True, timeout=10) as response:
                    return url, response.status
        except Exception as e:
            return url, 0
    
    async def fetch_pagespeed_data(self, session: aiohttp.ClientSession, url: str, 
                                   strategy: str = 'mobile') -> Dict[str, Any]:
        """Fetch PageSpeed metrics for a URL"""
        params = {
            'url': url,
            'key': self.api_key,
            'strategy': strategy
        }
        
        try:
            async with self.semaphore:
                async with session.get(API_ENDPOINT, params=params, timeout=30) as response:
                    if response.status != 200:
                        return {
                            'url': url,
                            'strategy': strategy,
                            'error': f"API Error: {response.status}",
                            'success': False
                        }
                    
                    data = await response.json()
                    lighthouse = data.get('lighthouseResult', {})
                    
                    # Extract performance metrics
                    performance = lighthouse.get('categories', {}).get('performance', {})
                    audits = lighthouse.get('audits', {})
                    
                    return {
                        'url': url,
                        'strategy': strategy,
                        'success': True,
                        'score': round((performance.get('score', 0) or 0) * 100, 1),
                        'firstContentfulPaint': audits.get('first-contentful-paint', {}).get('displayValue', 'N/A'),
                        'speedIndex': audits.get('speed-index', {}).get('displayValue', 'N/A'),
                        'timeToInteractive': audits.get('interactive', {}).get('displayValue', 'N/A'),
                        'firstMeaningfulPaint': audits.get('first-meaningful-paint', {}).get('displayValue', 'N/A'),
                        'largestContentfulPaint': audits.get('largest-contentful-paint', {}).get('displayValue', 'N/A'),
                        'totalBlockingTime': audits.get('total-blocking-time', {}).get('displayValue', 'N/A'),
                        'cumulativeLayoutShift': audits.get('cumulative-layout-shift', {}).get('displayValue', 'N/A'),
                    }
        except Exception as e:
            return {
                'url': url,
                'strategy': strategy,
                'error': str(e),
                'success': False
            }
    
    async def process_url_batch(self, urls: List[str]) -> pd.DataFrame:
        """Process a batch of URLs for both mobile and desktop strategies"""
        valid_urls = []
        results = []
        
        # Check URL validity first
        async with aiohttp.ClientSession() as session:
            status_tasks = [self.check_status_code(session, url.strip()) for url in urls if url.strip()]
            statuses = await asyncio.gather(*status_tasks)
            
            valid_urls = [url for url, status in statuses if status in (200, 301, 302)]
            
            # Create tasks for both mobile and desktop
            tasks = []
            for url in valid_urls:
                tasks.append(self.fetch_pagespeed_data(session, url, 'mobile'))
                tasks.append(self.fetch_pagespeed_data(session, url, 'desktop'))
            
            # Execute all tasks
            results = await asyncio.gather(*tasks)
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        return df

def run_analysis():
    """Main function to run the PageSpeed analysis"""
    st.set_page_config(page_title="PageSpeed Insights Analyzer", page_icon="🚀", layout="wide")
    
    st.title("PageSpeed Insights Analyzer")
    st.write("Analyze performance metrics for multiple URLs using Google PageSpeed Insights API")
    
    # API Key input (with option to use environment variable)
    custom_api_key = st.text_input("API Key (optional if set in environment)", 
                                   type="password", 
                                   help="Get your API key from Google Cloud Console")
    
    # URL input
    urls_input = st.text_area("Enter URLs (one per line)", 
                             height=150,
                             help="Enter full URLs including https://")
    
    # Run analysis button
    if st.button("Analyze URLs", type="primary"):
        if not urls_input.strip():
            st.error("Please enter at least one URL")
            return
        
        # Parse and clean URLs
        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        
        # Check for valid URLs
        valid_urls = []
        for url in urls:
            if not url.startswith(('http://', 'https://')):
                st.warning(f"URL '{url}' doesn't include protocol. Adding https://")
                url = f"https://{url}"
            valid_urls.append(url)
        
        if not valid_urls:
            st.error("No valid URLs found")
            return
        
        # Initialize analyzer
        analyzer = PageSpeedAnalyzer(api_key=custom_api_key if custom_api_key else None)
        
        # Show progress
        progress_bar = st.progress(0)
        status_message = st.empty()
        
        # Run asynchronous analysis
        status_message.text("Checking URL statuses...")
        
        # Run the async analysis in a separate thread
        with ThreadPoolExecutor() as executor:
            loop = asyncio.new_event_loop()
            
            async def run_analysis_task():
                return await analyzer.process_url_batch(valid_urls)
            
            future = executor.submit(lambda: loop.run_until_complete(run_analysis_task()))
            
            # Show progress update
            total_steps = len(valid_urls) * 2  # Mobile and desktop for each URL
            completed = 0
            while not future.done():
                time.sleep(0.5)
                completed += 1
                progress = min(completed / total_steps, 0.99)
                progress_bar.progress(progress)
                status_message.text(f"Analyzing URLs... ({completed}/{total_steps} requests)")
            
            # Get results
            results_df = future.result()
            st.write("--- DEBUG: Raw Results List ---", results_df)
        
        # Clear progress indicators
        progress_bar.progress(1.0)
        status_message.text("Analysis complete!")
        
        # Display results
        if len(results_df) == 0:
            st.warning("No results found. Check URLs and API key.")
            return
        
        # Split into mobile and desktop results
        if 'success' in results_df.columns:
            # Filter successful results
            results_df_filtered = results_df[results_df['success'] == True] # Temporarily use a new name
            st.write("--- DEBUG: Filtered DataFrame (Successful Only) ---", results_df_filtered) # <--- ADD THIS LINE (around line 136)
        else:
            results_df_filtered = pd.DataFrame() # Handle case where 'success' column might be missing
            st.write("--- DEBUG: 'success' column not found in results ---") # <-- ADD THIS LINE TOO
            # Filter successful results
            results_df = results_df[results_df['success'] == True]
        
        # Check the length of the *filtered* DataFrame
        if len(results_df_filtered) == 0:
            st.error("All requests failed. Check your API key and URLs. See details below.")
            # Show details of the original DataFrame that contained failures
            if 'success' in results_df.columns:
                failed_df = results_df[results_df['success'] == False]
                if not failed_df.empty:
                    st.write("--- DEBUG: Failed Request Details ---")
                    st.dataframe(failed_df[['url', 'strategy', 'error']]) # Display failures
                else:
                     st.write("--- DEBUG: No specific failures found, but successful results are empty. Original results:")
                     st.dataframe(results_df) # Show original if no failures marked but still empty after filter
            else:
                st.write("--- DEBUG: Could not filter by success. Original results:")
                st.dataframe(results_df) # Show original if filtering wasn't possible

            return # Stop execution here

        # IMPORTANT: If successful, continue with the filtered data
        results_df = results_df_filtered # Assign the filtered data back to results_df
        
        # Create separate dataframes for mobile and desktop
        mobile_df = results_df[results_df['strategy'] == 'mobile'].reset_index(drop=True)
        desktop_df = results_df[results_df['strategy'] == 'desktop'].reset_index(drop=True)
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["Mobile Performance", "Desktop Performance", "Comparison"])
        
        with tab1:
            st.header("Mobile Performance Results")
            st.dataframe(mobile_df.drop(['strategy', 'success'], axis=1, errors='ignore'))
            
            # Create score visualization
            if len(mobile_df) > 0:
                fig = px.bar(mobile_df, x='url', y='score', title='Mobile Performance Scores',
                            color='score', color_continuous_scale=['red', 'yellow', 'green'],
                            range_color=[0, 100])
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Desktop Performance Results")
            st.dataframe(desktop_df.drop(['strategy', 'success'], axis=1, errors='ignore'))
            
            # Create score visualization
            if len(desktop_df) > 0:
                fig = px.bar(desktop_df, x='url', y='score', title='Desktop Performance Scores',
                            color='score', color_continuous_scale=['red', 'yellow', 'green'],
                            range_color=[0, 100])
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Mobile vs Desktop Comparison")
            
            # Create comparison dataframe
            if len(mobile_df) > 0 and len(desktop_df) > 0:
                comparison_df = pd.DataFrame({
                    'URL': mobile_df['url'],
                    'Mobile Score': mobile_df['score'],
                    'Desktop Score': desktop_df['score'],
                    'Difference': desktop_df['score'] - mobile_df['score']
                })
                
                st.dataframe(comparison_df)
                
                # Plot comparison
                fig = px.bar(comparison_df, x='URL', y=['Mobile Score', 'Desktop Score'], 
                            barmode='group', title='Mobile vs Desktop Performance')
                st.plotly_chart(fig, use_container_width=True)
        
        # Option to download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="pagespeed_results.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    run_analysis()
