#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube Social Media Analysis Tool

This script implements a comprehensive tool for analyzing YouTube videos, including:
- Searching for videos by title
- Analyzing video metadata and engagement metrics
- Capturing and analyzing comments
- Detecting sentiment in comments and titles
- Identifying potential adversarial risks
- Generating summaries of findings
- Sending email notifications with analysis results

Date: June 18, 2025
"""

import os
import json
import re
import logging
from datetime import datetime
from collections import Counter

# Google API libraries
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

# Data analysis libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# NLP libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer



# Email libraries
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .init_application import initialization_result


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("youtube_analysis_tool")

# Download NLTK data for sentiment analysis
nltk.download('vader_lexicon', quiet=True)



# Import authentication and email functions
from .youtube_auth import get_authenticated_service
from .email_service import send_error_notification, send_consolidated_analysis_notification



def search_videos_by_title(youtube, query, max_results=10, batch_size=50, publish_after=None, publish_before=None, 
                      order="relevance", location=None, location_radius=None, region_code=None, relevance_language=None, include_shorts=False):
    """
    Search for YouTube videos by title query with pagination support and filters
    
    Args:
        youtube: Authenticated YouTube API client
        query: Search query string
        max_results: Maximum number of results to return
        batch_size: Number of results to retrieve per API request (max 50)
        publish_after: Optional RFC 3339 formatted date-time (e.g., '2023-01-01T00:00:00Z') for videos published after this date
        publish_before: Optional RFC 3339 formatted date-time (e.g., '2023-12-31T23:59:59Z') for videos published before this date
        order: Order of search results ('relevance', 'date', 'rating', 'viewCount', or 'title')
        location: Optional geographic location point (latitude,longitude) format
        location_radius: Optional radius from the location point (e.g., '5km', '10mi')
                         Required if location is specified
        region_code: Optional ISO 3166-1 alpha-2 country code (e.g., 'US', 'GB', 'IN')
                     Filter results to videos in a specific region
        relevance_language: Optional ISO 639-1 two-letter language code (e.g., 'en', 'es', 'fr')
                           Filter results to be most relevant to a specific language
        
    Returns:
        List of dictionaries containing video information
    """
    videos = []
    next_page_token = None
    next_page_token1 = None
    
    # Cap batch_size at 50 (YouTube API limit)
    batch_size = min(batch_size, 50)
    
    # Calculate how many total iterations we might need
    remaining_results = max_results
    
    # Process and validate the query
    search_query = query.strip()
    
    # Check if query exceeds YouTube's approximate limit
    if len(search_query) > 128:
        logger.warning(f"Query is {len(search_query)} characters long, which may exceed YouTube's limit.")
        logger.warning("YouTube typically supports queries up to ~128 characters. The query may be truncated.")
    
    # Detect advanced operators in the query
    has_or = '|' in search_query
    has_exclusion = any(term.startswith('-') for term in search_query.split())
    has_exact = '"' in search_query
    has_channel = 'channel:' in search_query.lower()
    has_intitle = 'intitle:' in search_query.lower()
    
    # Print search information
    logger.info(f"Searching for videos matching: '{search_query}'")
    
    # Print detected operators
    operators = []
    if has_or:
        operators.append("OR alternatives")
    if has_exclusion:
        operators.append("term exclusion")
    if has_exact:
        operators.append("exact phrases")
    if has_channel:
        operators.append("channel filter")
    if has_intitle:
        operators.append("title filter")
    
    if operators:
        logger.info(f"Detected query operators: {', '.join(operators)}")
    
    # Print date filter information
    if publish_after or publish_before:
        date_range_str = []
        if publish_after:
            date_range_str.append(f"after {publish_after}")
        if publish_before:
            date_range_str.append(f"before {publish_before}")
        logger.info(f"Date filters: {' and '.join(date_range_str)}")
    
    # Print location filter information
    if location:
        logger.info(f"Location filter: Coordinates {location} with radius {location_radius}")
    
    # Print region filter information
    if region_code:
        logger.info(f"Region filter: {region_code}")
    
    # Print language filter information
    if relevance_language:
        logger.info(f"Language relevance: {relevance_language}")
    
    while remaining_results > 0:
        # Determine how many results to request in this batch
        current_batch_size = min(batch_size, remaining_results)
        
        # Make the search request
        search_params = {
            'q': query,
            'part': 'id,snippet',
            'maxResults': current_batch_size,
            'type': 'video',
            'order': order
        }
        # YouTube API supports 'videoDuration' filter: 'any', 'short', 'medium', 'long'
        # Example: search_params['videoDuration'] = 'short'
        # Uncomment and set as needed, e.g.:
        # search_params['videoDuration'] = 'short'  # < 4 minutes
        # search_params['videoDuration'] = 'medium' # 4-20 minutes
        # search_params['videoDuration'] = 'long'   # > 20 minutes
        
        # Add date range filters if provided
        if publish_after:
            search_params['publishedAfter'] = publish_after
        if publish_before:
            search_params['publishedBefore'] = publish_before
            
        # Add location filters if provided (must provide both location and radius)
        if location and location_radius:
            search_params['location'] = location
            search_params['locationRadius'] = location_radius
            
        # Add region code filter if provided
        if region_code:
            search_params['regionCode'] = region_code
            
        # Add language relevance if provided
        if relevance_language:
            search_params['relevanceLanguage'] = relevance_language
        
        # Add page token if we have one from a previous request
        if next_page_token:
            search_params['pageToken'] = next_page_token

    
        # Perform the first search (no videoDuration filter)

        logger.info(f"Performing search with parameters: {search_params}")
        search_response = youtube.search().list(**search_params).execute()

        # Perform the second search with videoDuration='short'
        search_params_short = search_params.copy()
        search_params_short['videoDuration'] = 'short'
        if next_page_token1:
            search_params_short['pageToken'] = next_page_token1
        search_response1 = youtube.search().list(**search_params_short).execute()
        

        # Merge items from both responses, avoiding duplicates by videoId
        items_by_id = {}
        for resp in [search_response, search_response1]:
            for item in resp.get('items', []):
                #print(f"Processing item: {item}")
                if item['id']['kind'] == 'youtube#video':
                    video_id = item['id']['videoId']
                    items_by_id[video_id] = item  # Overwrite if duplicate, keeps latest



        # Process the merged results
        result_count = 0
        for search_result in items_by_id.values():
            if search_result['id']['kind'] == 'youtube#video':
                video_id = search_result['id']['videoId']
                title = search_result['snippet']['title']
                channel_title = search_result['snippet']['channelTitle']
                published_at = search_result['snippet']['publishedAt']

                # Get additional video details
                video_response = youtube.videos().list(
                    part='statistics,snippet,contentDetails',
                    id=video_id
                ).execute()

                if video_response['items']:
                    video_details = video_response['items'][0]
                    statistics = video_details['statistics']
                    content_details = video_details.get('contentDetails', {})
                    duration_str = content_details.get('duration', 'PT0S')
                    # Parse ISO 8601 duration (e.g., PT59S, PT1M, PT1M1S)
                    import isodate
                    try:
                        duration_seconds = int(isodate.parse_duration(duration_str).total_seconds())
                    except Exception:
                        duration_seconds = 0

                    # Shorts: duration <= 60 seconds
                    is_short = duration_seconds <= 60

                    if include_shorts and not is_short:
                        continue  # Skip non-shorts if only shorts requested

                    videos.append({
                        'id': video_id,
                        'title': title,
                        'channel': channel_title,
                        'published_at': published_at,
                        'view_count': statistics.get('viewCount', '0'),
                        'like_count': statistics.get('likeCount', '0'),
                        'comment_count': statistics.get('commentCount', '0'),
                        'description': video_details['snippet'].get('description', ''),
                        'url': f'https://youtube.com/watch?v={video_id}',
                        'duration_seconds': duration_seconds,
                        'is_short': is_short
                    })
                    result_count += 1
        
        # Update remaining results counter
        remaining_results -= result_count
        
        # Check if we have more pages of results
        next_page_token = search_response.get('nextPageToken')
        next_page_token1 = search_response1.get('nextPageToken')

        # Break if we've collected enough results or there are no more pages
        if not next_page_token or result_count == 0:
            break
            
        logger.info(f"Retrieved {len(videos)} videos so far, fetching more...")
    
    logger.info(f"Found a total of {len(videos)} videos matching '{query}'")
    return videos


def get_video_comments(youtube, video_id, max_results=100):
    """
    Retrieve comments for a specific YouTube video
    
    Args:
        youtube: Authenticated YouTube API client
        video_id: YouTube video ID
        max_results: Maximum number of comments to retrieve
        
    Returns:
        List of dictionaries containing comment information
    """
    comments = []
    next_page_token = None
    
    # First check if comments are enabled for this video
    try:
        logger.info(f"Checking video details for {video_id}...")
        video_response = youtube.videos().list(
            part="statistics",
            id=video_id
        ).execute()
        
        if not video_response['items']:
            logger.warning(f"Video {video_id} not found")
            return comments
            
        statistics = video_response['items'][0]['statistics']
        if 'commentCount' not in statistics or int(statistics['commentCount']) == 0:
            logger.info(f"No comments available for video {video_id} (comments might be disabled)")
            return comments
            
        logger.info(f"Video has approximately {statistics.get('commentCount', '0')} comments")
        
    except Exception as e:
        logger.error(f"Error checking video statistics: {e}")
    
    # Now try to fetch the comments
    try:
        logger.info(f"Beginning comment retrieval for video {video_id}")
        
        while len(comments) < max_results:
            logger.info(f"Fetching batch of comments (have {len(comments)} so far, target: {max_results})...")
            
            # Set up request parameters
            request_params = {
                'part': 'snippet',
                'videoId': video_id,
                'maxResults': min(100, max_results - len(comments)),
                'textFormat': 'plainText'
            }
            
            if next_page_token:
                request_params['pageToken'] = next_page_token
                
            # Make the API request
            comment_response = youtube.commentThreads().list(**request_params).execute()
            
            # Check if we got any comments back
            items = comment_response.get('items', [])
            if not items:
                logger.info("No comments returned in this batch")
                break
                
            logger.info(f"Retrieved {len(items)} comments in this batch")
            
            # Extract comment data
            for item in items:
                try:
                    comment = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'author': comment['authorDisplayName'],
                        'text': comment['textDisplay'],
                        'like_count': comment['likeCount'],
                        'published_at': comment['publishedAt']
                    })
                except KeyError as ke:
                    logger.warning(f"Missing expected field in comment data: {ke}")
                
            # Check if there are more comments
            next_page_token = comment_response.get('nextPageToken')
            if not next_page_token or len(comments) >= max_results:
                break
                
        logger.info(f"Successfully retrieved {len(comments)} comments")
            
    except Exception as e:
        logger.error(f"Error retrieving comments: {e}")
        logger.error(f"This could be due to:")
        logger.error("  - Comments being disabled for the video")
        logger.error("  - Insufficient permissions in the OAuth scopes")
        logger.error("  - Issues with the credentials")
        logger.error("  - YouTube API quota limits")
        logger.error("\nTip: Delete token.pickle and try again to force re-authentication")
    
    return comments


def comments_to_dataframe(comments):
    """Convert comments list to pandas DataFrame with cleaned text"""
    if not comments:
        return pd.DataFrame(columns=['author', 'text', 'like_count', 'published_at'])
    
    df = pd.DataFrame(comments)
    
    # Clean HTML from comment text
    df['text_clean'] = df['text'].apply(lambda x: re.sub('<.*?>', '', x))
    
    # Convert published_at to datetime
    df['published_at'] = pd.to_datetime(df['published_at'])
    
    return df


def analyze_sentiment(text):
    """
    Analyze sentiment of text using VADER sentiment analyzer
    
    Args:
        text: Text string to analyze
        
    Returns:
        Tuple of (sentiment_category, sentiment_scores)
    """
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    # Determine sentiment category
    if sentiment_scores['compound'] >= 0.05:
        return 'positive', sentiment_scores
    elif sentiment_scores['compound'] <= -0.05:
        return 'negative', sentiment_scores
    else:
        return 'neutral', sentiment_scores


def analyze_comments_sentiment(comments_df):
    """
    Analyze sentiment of comments in a DataFrame
    
    Args:
        comments_df: DataFrame with comments
        
    Returns:
        DataFrame with sentiment analysis and summary dictionary
    """
    if comments_df.empty:
        return comments_df, {"positive": 0, "neutral": 0, "negative": 0}
    
    # Apply sentiment analysis to each comment
    sentiment_results = comments_df['text_clean'].apply(analyze_sentiment)
    
    # Split results into separate columns
    comments_df['sentiment'] = sentiment_results.apply(lambda x: x[0])
    comments_df['sentiment_scores'] = sentiment_results.apply(lambda x: x[1])
    comments_df['compound_score'] = comments_df['sentiment_scores'].apply(lambda x: x['compound'])
    
    # Calculate sentiment distribution
    sentiment_counts = comments_df['sentiment'].value_counts().to_dict()
    
    # Ensure all categories exist
    for category in ['positive', 'neutral', 'negative']:
        if category not in sentiment_counts:
            sentiment_counts[category] = 0
    
    return comments_df, sentiment_counts




def detect_adversarial_risks(video_data, comments_df=None,transcript=None, use_llm=True, llm_model="gpt-4o", api_key=None):
    """
    Detect potential adversarial risks in video metadata and comments
    
    Args:
        video_data: Dictionary of video metadata
        comments_df: DataFrame with comments and sentiment analysis
        use_llm: Whether to use LLM for advanced adversarial analysis
        llm_model: The model to use for LLM analysis
        api_key: API key for the LLM provider (if None, uses environment variable)
        
    Returns:
        Dictionary with risk assessment results
    """

    risks = []
    # Load risk_keywords and ignore_words from initialization_result["env_vars"]
    env_vars = initialization_result["env_vars"]
    # Expecting comma-separated values in .env
    risk_keywords = [kw.strip() for kw in env_vars.get("RISK_KEYWORDS", "").split(",") if kw.strip()]
    ignore_words = [iw.strip() for iw in env_vars.get("IGNORE_WORDS", "").split(",") if iw.strip()]

    def contains_ignore_word(text):
        return any(ignore_word.lower() in text.lower() for ignore_word in ignore_words)

    title_text = video_data['title']
    desc_text = video_data['description']

    # Only check for risks if ignore words are NOT present
    #print("AAA",contains_ignore_word(title_text))
    ignore_flag=contains_ignore_word(title_text)
    if not ignore_flag:
        title_risks = [kw for kw in risk_keywords if re.search(r'\b' + kw + r'\b', title_text.lower())]
    else:
        title_risks = []

    if not contains_ignore_word(desc_text):
        desc_risks = [kw for kw in risk_keywords if re.search(r'\b' + kw + r'\b', desc_text.lower())]
    else:
        desc_risks = []

    # # Check specifically for security exploitation patterns in title
    security_risks = ['hack', 'exploit', 'vulnerability', 'breach', 'bypass', 'jailbreak', 'root']
    # security_pattern = r'\bhow\s+to\s+(?:hack|exploit|bypass|jailbreak|root|compromise)\b.*?\b(?:device|phone|system|app|miko|robot)\b'
    # has_security_exploit_pattern = bool(re.search(security_pattern, title_text.lower())) and not contains_ignore_word(title_text)

    # # Detect if title mentions specific devices being hacked
    # device_hack_pattern = r'\bhack\s+[a-z0-9]+(?:\s+[0-9])?(?:\s+with)?\b'
    # has_device_hack = bool(re.search(device_hack_pattern, title_text.lower())) and not contains_ignore_word(title_text)
    if transcript:
        logger.debug("Title Risks:", title_text,title_risks, )
    else:
        logger.debug("Title Risks 2:", title_text,title_risks, )

    #print("Title Risks:", title_text,title_risks, )

    if ignore_flag:
        
        return {
            'risks_detected': len(risks) > 0,
            'risk_level': "low",
            'details': risks
        }
    
    if not transcript:
        if title_risks :
            # Automatically high severity for security-related risks
            security_related = any(
                any(risk.lower() == sr.lower() for sr in security_risks)
                for risk in title_risks
            )
            risks.append({
                'source': 'title',
                'keywords': title_risks,
                'severity': 'high' if security_related else ('high' if len(title_risks) > 1 else 'medium'),
                'analysis_method': 'keyword',
                #'security_exploit_detected': has_security_exploit_pattern or has_device_hack
            })
            
        if desc_risks:
            risks.append({
                'source': 'description',
                'keywords': desc_risks,
                'severity': 'medium',
                'analysis_method': 'keyword'
            })
    
    # Use LLM for advanced adversarial analysis if requested
    if use_llm:
        try:
            import litellm
            import hashlib
            # Configure LiteLLM with API key if provided
            if api_key:
                litellm.api_key = initialization_result["env_vars"]['LLM_API_KEY']

            # Get product names and risk keywords from env
            env_vars = initialization_result["env_vars"]
            product_names = [p.strip() for p in env_vars.get("PRODUCT_NAMES", "").split(",") if p.strip()]
            product_names_str = ", ".join([f'\"{p}\"' for p in product_names])
            risk_keywords = [kw.strip() for kw in env_vars.get("RISK_KEYWORDS", "hacking,exploits,vulnerabilities,bypassing security").split(",") if kw.strip()]
            risk_keywords_str = ", ".join([f'\"{kw}\"' for kw in risk_keywords])
            exclusions = [e.strip() for e in env_vars.get("EXCLUSIONS", "").split(",") if e.strip()]
            exclusions_prompt = ""
            if exclusions:
                exclusions_prompt = "however strictly ignore the videos with the title or channels which are related to youtuber or youtube account\n" + "\n".join([f"- {e}" for e in exclusions])

            # Create content for LLM analysis
            content = f"Title: {video_data['title']}\n\nDescription: {video_data['description']}"
            if transcript:
                content += f"\n\nTranscript: {transcript}"

            # Sample negative comments if available
            if comments_df and not comments_df.empty and 'sentiment' in comments_df.columns:
                negative_comments = comments_df[comments_df['sentiment'] == 'negative']
                if not negative_comments.empty:
                    sample_comments = negative_comments.sample(min(5, len(negative_comments)))['text_clean'].tolist()
                    content += "\n\nSample Comments:\n" + "\n".join([f"- {comment}" for comment in sample_comments])

            # Prompt for LLM
            prompt = f"""
            Analyze the following YouTube video content for potential adversarial risks, harmful content, 
            and problematic themes. Look for subtle indicators that might be missed by simple 
            keyword analysis. Only content referencing {risk_keywords_str} 
            or similar terms in conjunction with only the products or devices [{product_names_str}] MUST be categorized 
            as adversarial risks.

            {content}
            
            Assess the content for the following risk categories:

            1. Security vulnerability exploitation (HIGHEST PRIORITY)
            2. Hacking tutorials or guides (HIGHEST PRIORITY)
            3. Privacy violations or threats
            4. Unauthorized access instructions
            5. Malware distribution or promotion
            6. Device exploitation techniques
            
            {exclusions_prompt}
            
            For each category you identify as potentially problematic, provide:
            - The category of concern
            - The severity level (low, medium, high)
            - Brief explanation with specific examples from the content
            - Confidence level in your assessment (low, medium, high)
            - Evidence: specific text from content
            
            Format your response as a JSON object with this structure:
            {{
                "risks_detected": boolean,
                "risk_categories": [
                    {{
                        "category": "category name",
                        "severity": "low|medium|high",
                        "explanation": "brief explanation",
                        "confidence": "low|medium|high",
                        "evidence": "specific text from content"
                    }}
                ],
                "overall_risk_level": "low|medium|high",
                "summary": "brief summary of findings"
            }}
            
            If no risks are detected, return an appropriate JSON response.
            """

            # --- Caching logic ---
            video_id = video_data.get('id', 'unknown')
            prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
            output_dir = os.path.join("output", video_id)
            os.makedirs(output_dir, exist_ok=True)
            cache_file = os.path.join(output_dir, f"{prompt_hash}_{video_id}.json")

            if os.path.exists(cache_file):
                logger.debug(f"    → Using cached LLM output: {cache_file}")
                with open(cache_file, "r", encoding="utf-8") as f:
                    llm_analysis = json.load(f)
            else:
                logger.debug(f"    → Performing advanced adversarial analysis using LLM ({llm_model})...")
                response = litellm.completion(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1024
                )

                # Extract the LLM response content
                content = response.choices[0].message.content

                # Clean the JSON response from LLM
                def clean_json_response(response_text):
                    cleaned = re.sub(r'```(?:json)?\s*', '', response_text)
                    cleaned = re.sub(r'\s*```', '', cleaned)
                    cleaned = re.sub(r'<.*?>', '', cleaned)
                    cleaned = cleaned.replace('\\"', '"')
                    cleaned = cleaned.replace('\\n', ' ')
                    cleaned = cleaned.strip()
                    json_match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
                    if json_match:
                        cleaned = json_match.group(1)
                    return cleaned

                cleaned_content = clean_json_response(content)
                logger.debug(f"    → Cleaned LLM response for JSON parsing ",cleaned_content)
                try:
                    llm_analysis = json.loads(cleaned_content)
                    # Save to cache
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(llm_analysis, f, indent=2)
                except json.JSONDecodeError:
                    logger.error(f"    ⚠️ Could not parse LLM response as JSON")
                    risks.append({
                        'source': 'llm_error',
                        'issue': 'parsing_error',
                        'severity': 'low',
                        'analysis_method': 'llm'
                    })
                    llm_analysis = None

            # Add LLM findings to risks if any were detected
            if llm_analysis and llm_analysis.get('risks_detected', False):
                for risk in llm_analysis.get('risk_categories', []):
                    if risk.get('confidence') != 'low':
                        risks.append({
                            'source': 'llm_analysis',
                            'category': risk.get('category'),
                            'explanation': risk.get('explanation'),
                            'severity': risk.get('severity', 'medium'),
                            'evidence': risk.get('evidence'),
                            'confidence': risk.get('confidence'),
                            'summary': llm_analysis.get('summary'),
                            'analysis_method': 'llm'
                        })
                logger.debug(f"    ✓ LLM analysis complete")

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"    ⚠️ Error during LLM analysis: {e}")
            risks.append({
                'source': 'llm_error',
                'issue': str(e),
                'severity': 'low',
                'analysis_method': 'llm'
            })
    
    # Check for risk patterns in comments (if we have comments)
    if comments_df and not comments_df.empty and 'sentiment' in comments_df.columns:
        # High ratio of negative comments
        negative_comments = comments_df[comments_df['sentiment'] == 'negative']
        if len(negative_comments) > 0.3 * len(comments_df):
            risks.append({
                'source': 'comments',
                'issue': 'high_negative_ratio',
                'severity': 'medium',
                'percentage': len(negative_comments) / len(comments_df) * 100,
                'analysis_method': 'statistical'
            })
        
        # Look for risk keywords in comments
        comment_risks = []
        for _, comment in negative_comments.iterrows():
            found_keywords = [kw for kw in risk_keywords if re.search(r'\b' + kw + r'\b', 
                                                                     comment['text_clean'].lower())]
            if found_keywords:
                comment_risks.append({
                    'keywords': found_keywords,
                    'comment_text': comment['text_clean'][:100] + '...' if len(comment['text_clean']) > 100 else comment['text_clean']
                })
        
        if comment_risks:
            risks.append({
                'source': 'comments',
                'issue': 'risk_keywords',
                'severity': 'high' if len(comment_risks) > 5 else 'medium',
                'examples': comment_risks[:5],  # Include up to 5 examples
                'analysis_method': 'keyword'
            })
    
    # Determine overall risk level
    # Gather severity from different sources
    title_desc_risk = None
    llm_risk = None

    # Get product names from env_vars
    env_vars = initialization_result["env_vars"]
    product_names = [p.strip().lower() for p in env_vars.get("PRODUCT_NAMES", "").split(",") if p.strip()]

    # Check for title/description risk and LLM risk
    title_risk = None
    description_risk = None
    for r in risks:
        if r.get('source') == 'title':
            if r.get('severity') == 'high':
                title_risk = 'high'
            elif r.get('severity') == 'medium':
                title_risk = 'medium'
        if r.get('source') == 'description':
            if r.get('severity') == 'high':
                description_risk = 'high'
            elif r.get('severity') == 'medium':
                description_risk = 'medium'
        if r.get('source') == 'llm_analysis':
            llm_risk = r.get('severity')
    # For backward compatibility, set title_desc_risk as the highest of title/description
    title_desc_risk = None
    if title_risk == 'high' or description_risk == 'high':
        title_desc_risk = 'high'
    elif title_risk == 'medium' or description_risk == 'medium':
        title_desc_risk = 'medium'

    # Check if title or description contains any product name
    def contains_product_name(text):
        return any(pn in text.lower() for pn in product_names)

    # Logic for risk level
    if llm_risk == 'high':
        risk_level = 'high'
    elif (title_risk=='high') and (description_risk =='medium' or description_risk == 'high') and (llm_risk is None or llm_risk == 'low'):
        # Check if product name is present in title or description
        if contains_product_name(video_data.get('title', '')) or contains_product_name(video_data.get('description', '')):
            risk_level = 'high'
        else:
            risk_level = 'medium'
    else:
        risk_level = 'low'

    # elif any(r.get('severity') == 'medium' for r in risks):
    #     risk_level = 'medium'
    # elif any(r.get('severity') == 'high' for r in risks):
    #     risk_level = 'high'
    # else:
    #     risk_level = 'low'
    
    return {
        'risks_detected': len(risks) > 0,
        'risk_level': risk_level,
        'details': risks
    }


def get_recommendation(sentiment_data, risk_analysis):
    """
    Generate a recommendation based on analysis results
    """
    if risk_analysis['risk_level'] == 'high':
        return "ATTENTION REQUIRED: This video has significant risks that require immediate review."
    elif risk_analysis['risk_level'] == 'medium':
        return "CAUTION: This video has some concerning elements that should be monitored."
    # elif sentiment_data.get('negative', 0) > sentiment_data.get('positive', 0):
    #     return "REVIEW SUGGESTED: While no major risks were detected, negative sentiment is prevalent."
    else:
        return "NO ACTION NEEDED: This video appears to be safe with neutral to positive sentiment."


def generate_summary(video_data, comments_df, sentiment_data, risk_analysis):
    """
    Generate a comprehensive summary of video analysis
    
    Args:
        video_data: Dictionary of video metadata
        comments_df: DataFrame with comments and sentiment analysis
        sentiment_data: Dictionary with sentiment distribution
        risk_analysis: Dictionary with risk assessment
        
    Returns:
        Dictionary with analysis summary
    """
    # Calculate engagement metrics
    try:
        view_count = int(video_data['view_count'])
        like_count = int(video_data['like_count']) if 'like_count' in video_data else 0
        comment_count = int(video_data['comment_count']) if 'comment_count' in video_data else 0
        
        engagement_rate = ((like_count + comment_count) / view_count * 100) if view_count > 0 else 0
    except (ValueError, KeyError):
        view_count = 0
        like_count = 0
        comment_count = 0
        engagement_rate = 0

    logger.debug(f"Risk Analysis: {risk_analysis}")
    # Generate the summary
    summary = {
        'video': {
            'title': video_data['title'],
            'url': video_data['url'],
            'channel': video_data['channel'],
            'published_at': video_data['published_at'],
            'view_count': view_count,
            'like_count': like_count,
            'comment_count': comment_count,
            'engagement_rate': engagement_rate
        },
        'sentiment_analysis': {
            # 'overall_sentiment': max(sentiment_data.items(), key=lambda x: x[1])[0] if sentiment_data else 'unknown',
            # 'distribution': sentiment_data
        },
        'risk_assessment': {
            'risk_level': risk_analysis['risk_level'],
            'risks_detected': risk_analysis['risks_detected'],
            'summary': [f"{risk['source']}: {risk['severity']} risk" for risk in risk_analysis['details']]
        },
        'recommendation': get_recommendation(sentiment_data, risk_analysis),
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    # Add top negative comments if any exist
    if comments_df and not comments_df.empty and 'sentiment' in comments_df.columns:
        negative_comments = comments_df[comments_df['sentiment'] == 'negative']
        if not negative_comments.empty:
            summary['top_negative_comments'] = negative_comments.sort_values('compound_score', ascending=True)['text_clean'].head(3).tolist()
    
    return summary


def print_text_summary(summary):
    """
    Print a formatted text summary of the analysis
    """
    logger.debug(f"=== YOUTUBE VIDEO ANALYSIS SUMMARY ===")
    logger.debug(f"Title: {summary['video']['title']}")
    logger.debug(f"URL: {summary['video']['url']}")
    logger.debug(f"Channel: {summary['video']['channel']}")
    logger.debug(f"Published: {summary['video']['published_at']}")
    logger.debug(f"\nENGAGEMENT:")
    logger.debug(f"Views: {summary['video']['view_count']:,}")
    logger.debug(f"Likes: {summary['video']['like_count']:,}")
    logger.debug(f"Comments: {summary['video']['comment_count']:,}")
    logger.debug(f"Engagement Rate: {summary['video']['engagement_rate']:.2f}%")
    
    logger.debug(f"\nSENTIMENT ANALYSIS:")
    logger.debug(f"Overall Sentiment: {summary['sentiment_analysis']['overall_sentiment'].upper()}")
    for sentiment, count in summary['sentiment_analysis']['distribution'].items():
        total = sum(summary['sentiment_analysis']['distribution'].values())
        percentage = (count / total * 100) if total > 0 else 0
        logger.debug(f"- {sentiment}: {count} ({percentage:.1f}%)")
    
    logger.debug(f"\nRISK ASSESSMENT:")
    logger.debug(f"Risk Level: {summary['risk_assessment']['risk_level'].upper()}")
    if summary['risk_assessment']['risks_detected']:
        for risk in summary['risk_assessment']['summary']:
            logger.debug(f"- {risk}")
    else:
        logger.debug("- No significant risks detected")
    
    logger.debug(f"\nRECOMMENDATION:")
    logger.debug(summary['recommendation'])
    
    if 'top_negative_comments' in summary and summary['top_negative_comments']:
        logger.debug(f"\nSAMPLE NEGATIVE COMMENTS:")
        for i, comment in enumerate(summary['top_negative_comments'], 1):
            logger.debug(f"{i}. \"{comment[:100]}...\"" if len(comment) > 100 else f"{i}. \"{comment}\"")

    logger.debug(f"\nAnalysis completed at: {summary['analysis_timestamp']}")
    logger.debug("="*40)


# Using send_analysis_notification from email_service module


# Using send_consolidated_analysis_notification from email_service module


def download_first_60_seconds(youtube_url, output_file="first60.m4a"):
    import os
    import yt_dlp

    # Extract video_id from the YouTube URL
    import re
    m = re.search(r"v=([\w-]+)", youtube_url)
    video_id = m.group(1) if m else "unknown"
    output_dir = os.path.join("output", video_id)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, os.path.basename(output_file))

    # Check if file exists and is not zero size and is suitable for 1 minute audio (e.g., >100KB)
    min_expected_size = 100 * 1024  # 100 KB, adjust as needed for your use case
    if os.path.exists(output_file_path):
        file_size = os.path.getsize(output_file_path)
        if file_size > min_expected_size:
            logger.debug(f"File '{output_file_path}' already exists and is {file_size/1024:.1f} KB, skipping download.")
            return output_file_path
        else:
            logger.debug(f"File '{output_file_path}' exists but is too small ({file_size} bytes), will re-download.")

    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best/fastest',  # Try to get the fastest available audio format
        'outtmpl': output_file_path,
        'download_sections': ['*00:00-01:00'],  # first 60 seconds
        'postprocessor_args': [
            '-ss', '00:00:00',
            '-t', '00:01:00'
        ],
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
        'quiet': False,
        'noplaylist': True,
        'concurrent_fragment_downloads': 4,  # yt-dlp option to speed up downloads
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_file_path

import litellm

def transcribe_audio(file_path, model="whisper-1"):
    """
    Transcribe audio using LiteLLM's OpenAI Whisper API wrapper.

    Args:
        file_path (str): Path to the audio file.
        model (str): Whisper model to use (default: "whisper-1").

    Returns:
        str: Transcribed text.
    """
    logger.debug(f"\n🎧 Uploading {file_path} to Whisper API (via LiteLLM) for transcription...\n")
    litellm.api_key = initialization_result["env_vars"]['OPENAI_API_KEY']

    with open(file_path, "rb") as audio_file:
        transcript = litellm.transcription(
            model=model,
            file=audio_file
        )
    return transcript['text']

def save_transcript(text, filename="transcript.txt"):
    # Save transcript in the correct output/video_id directory
    import os
    video_id = filename.split("_")[0]
    output_dir = os.path.join("output", video_id)
    os.makedirs(output_dir, exist_ok=True)
    transcript_path = os.path.join(output_dir, os.path.basename(filename))
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.debug(f"\n✅ Transcript saved to {transcript_path}")
    return transcript_path

# Using send_error_notification imported from email_service module
    

    


def download_youtube_captions(video_id, youtube=None, language_code="en", duration="60"):
    """
    Download captions for a YouTube video using the YouTube Transcript API.
    This method doesn't require authentication with YouTube Data API.
    
    Args:
        video_id (str): The ID of the YouTube video.
        youtube: Not used, kept for backwards compatibility.
        language_code (str): Language code for the captions (default: "en" for English).
        use_alternative_method (bool): Whether to use the alternative method to fetch captions.
        
    Returns:
        dict: A dictionary with caption data and metadata, or None if no captions are available.
    """
    try:
        logger.debug(f"Fetching transcript for video: {video_id}")
        transcript=None
        output_dir = os.path.join("output", video_id)
        os.makedirs(output_dir, exist_ok=True)
        transcript_file = os.path.join(output_dir, video_id +"_"+duration+"_transcript.txt")
        # Check if transcript file exists and is not empty
        if os.path.exists(transcript_file) and os.path.getsize(transcript_file) > 0:
            logger.debug(f"Transcript file '{transcript_file}' already exists. Reading transcript from file.")
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript = f.read()
            logger.debug(f"Loaded transcript from '{transcript_file}'")
        else:
            youtube_link = f"https://www.youtube.com/watch?v={video_id}"
            output_file = video_id + "_first60.m4a"
            audio_path = download_first_60_seconds(youtube_link, output_file)
            transcript = transcribe_audio(audio_path)
            logger.debug(f"Transcription completed for video: {video_id}")
            logger.debug(f"Transcription result: {transcript}")
            save_transcript(transcript, transcript_file)

        return transcript
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error downloading captions: {e}")
        logger.error(error_details)

        # Create error information for notification
        error_info = {
            'video_id': video_id,
            'error_message': str(e),
            'error_details': error_details,
            'timestamp': datetime.now().isoformat(),
            'operation': 'Caption Download',
            'file_requested': transcript_file,
            'youtube_link': f"https://www.youtube.com/watch?v={video_id}"
        }
        
        # Send error notification email if recipient email is available
        from functools import partial
        send_error_notification_if_email = partial(send_error_notification, error_info=error_info)
        
        # The actual email sending will happen where the function is called with a recipient email
        return {"error": True, "error_info": error_info, "send_notification": send_error_notification_if_email}



def analyze_youtube_video_by_title(query, recipient_email=None, max_videos=3, save_results=True, 
                           publish_after=None, publish_before=None, order="relevance",
                           location=None, location_radius=None, region_code=None, relevance_language=None):
    """
    Main function to analyze YouTube videos by title search
    
    Args:
        query: Search query string
        recipient_email: Email address to send notifications to (optional)
        max_videos: Maximum number of videos to analyze
        save_results: Whether to save results to a JSON file
        publish_after: Optional RFC 3339 formatted date-time for videos published after this date
        publish_before: Optional RFC 3339 formatted date-time for videos published before this date
        order: Order of search results ('relevance', 'date', 'rating', 'viewCount', or 'title')
        location: Optional geographic location point (latitude,longitude) format
        location_radius: Optional radius from the location point (e.g., '5km', '10mi')
        region_code: Optional ISO 3166-1 alpha-2 country code (e.g., 'US', 'GB', 'IN')
        relevance_language: Optional ISO 639-1 two-letter language code (e.g., 'en', 'es', 'fr')
        
    Returns:
        List of dictionaries containing analysis summaries
    """
    try:
        # Initialize the YouTube API client
        youtube,error_info = get_authenticated_service()
        if youtube is None:
            send_error_notification(recipient_email, error_info)
            return [] 

        logger.debug(f"🔍 Analyzing search: '{query}'")

        # Detect if this is a multi-query (containing pipe character)
        is_multi_query = '|' in query
        queries=["|"]
        max_videos1=max_videos
        if is_multi_query:
            queries = [q.strip() for q in query.split('|') if q.strip()]
            logger.debug(f"Detected multiple search queries separated by '|'. Query count: {len(queries)}")
            logger.debug("Each query will be executed separately and results will be combined")
            max_videos1=max_videos*len(queries)
            logger.debug(f"Adjusted max_videos to: {max_videos} : {len(queries)}")
            #max_videos = max_videos // len(queries)
        # Use the multi-query search function (which handles both single and multiple queries)
        videos = search_videos_by_multiple_queries(
            youtube, 
            query,
            max_results=max_videos,  # Adjust max_results per query
            batch_size=50,
            publish_after=publish_after,
            publish_before=publish_before,
            order=order,
            location=location,
            location_radius=location_radius,
            region_code=region_code,
            relevance_language=relevance_language
        )
        
        if not videos:
            logger.debug("❌ No videos found matching the query.")
            return []

        logger.debug(f"✅ Found {len(videos)} videos. Analyzing top {min(max_videos1, len(videos))}...")
        #print(videos)
        
        results = []
        for i, video in enumerate(videos[:max_videos1], 1):
            logger.debug(f"\n📊 [{i}/{min(max_videos1, len(videos))}] Analyzing video: {video['title']}")

            # Detect potential risks
            logger.debug(f"    → Detecting potential risks...")
            risk_analysis = detect_adversarial_risks(video, None)
            logger.debug(f"    ✓ Risk detection complete. Risk level: {risk_analysis['risk_level'].upper()}")

            # If risk level is low, get transcript/captions
            transcript = None
            if risk_analysis['risk_level'] != 'low':
                logger.debug(f"    → Risk is low, attempting to fetch transcript/captions...")
                result = download_youtube_captions(video['id'])
                
                # Check if result contains error information
                if isinstance(result, dict) and result.get("error"):
                    logger.error("Error encountered while downloading captions!")
                    # Send error notification if recipient email is provided
                    if recipient_email:
                        error_notification_func = result.get("send_notification")
                        if error_notification_func:
                            error_notification_func(recipient_email)
                            logger.debug(f"Error notification sent to {recipient_email}")
                            transcript = None  # No transcript available due to error
                else:
                    # Normal processing if no error occurred
                    transcript = result
                    if transcript:
                        risk_analysis = detect_adversarial_risks(video, transcript=transcript)
                        logger.debug(f"    ✓ Risk2 detection complete. Risk level: {risk_analysis['risk_level'].upper()}")

            # Generate summary
            summary = generate_summary(video, None, None, risk_analysis)
            if transcript:
                summary['transcript'] = transcript
            results.append(summary)
            
            # Print text summary
            #print_text_summary(summary)
            
            # # Create visualizations if we have comments
            # if not comments_with_sentiment.empty:
            #     try:
            #         fig1 = plot_sentiment_distribution(sentiment_distribution)
            #         fig2 = plot_sentiment_over_time(comments_with_sentiment)
            #         fig1.show()
            #         fig2.show()
            #     except Exception as e:
            #         print(f"    ⚠️ Error creating visualizations: {e}")
            
            #Send email notification if requested
            # if recipient_email and risk_analysis['risk_level'] in ['high', 'medium']:
            #     print(f"    → Sending notification to {recipient_email}...")
            #     send_email_notification(recipient_email, summary)
        
        # Save results to JSON file in output/results directory
        if save_results:
            results_dir = os.path.join("output", "results")
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(results_dir, f"youtube_analysis_{timestamp}.json")
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            logger.debug(f"\n💾 Analysis results saved to {filename}")
        


        return results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"❌ Error during analysis: {e}")
        return []


def search_videos_by_multiple_queries(youtube, query_string, max_results=10, batch_size=50, **kwargs):
    """
    Search for YouTube videos by multiple queries separated by '|' and accumulate results
    
    Args:
        youtube: Authenticated YouTube API client
        query_string: Search query string potentially containing multiple queries separated by '|'
        max_results: Maximum number of results to return PER QUERY
        batch_size: Number of results to retrieve per API request (max 50)
        **kwargs: Additional keyword arguments to pass to search_videos_by_title
        
    Returns:
        List of dictionaries containing video information from all queries
    """
    # Check if query contains the multi-query delimiter
    if '|' not in query_string:
        # If not a multi-query, just do a normal search
        return search_videos_by_title(youtube, query_string, max_results, batch_size, **kwargs)
    
    # Split the query and strip whitespace
    queries = [q.strip() for q in query_string.split('|')]
    logger.debug(f"Multiple search queries detected. Will search for {len(queries)} different queries:")
    for i, q in enumerate(queries, 1):
        logger.debug(f"  Query {i}: '{q}'")


    # Accumulated results from all queries
    all_videos = []
    video_ids_seen = set()  # Track video IDs to avoid duplicates
    
    # Process each query
    for i, query in enumerate(queries, 1):
        if not query:  # Skip empty queries
            continue

        logger.debug(f"\nPerforming search {i}/{len(queries)}: '{query}'")

        # Search for videos with this query
        videos = search_videos_by_title(youtube, query, max_results, batch_size, **kwargs)
        
        # Track how many new videos we found
        new_videos_count = 0
        
        # Add unique videos to our accumulated results
        for video in videos:
            if video['id'] not in video_ids_seen:
                all_videos.append(video)
                video_ids_seen.add(video['id'])
                new_videos_count += 1

        logger.debug(f"Found {len(videos)} videos for query '{query}', "
                     f"{new_videos_count} were unique and added to results.")

    logger.debug(f"\nTotal unique videos found across all queries: {len(all_videos)}")

    return all_videos


def get_query_info():
    """Provide information about YouTube search query capabilities and formatting"""
    logger.debug("\n=== YOUTUBE SEARCH QUERY GUIDE ===")
    logger.debug("YouTube search supports various query formats and operators:")

    logger.debug("\n1. Multiple Terms and Phrases:")
    logger.debug("   - Simple words: tesla cybertruck")
    logger.debug("   - Exact phrases: \"artificial intelligence\"")
    logger.debug("   - Multiple sentences: how to make pancakes. best recipe.")

    logger.debug("\n2. Advanced Operators:")
    logger.debug("   - Standard OR operator: skateboarding | surfing")
    logger.debug("   - Multi-query searches: python tutorials | machine learning | data science")
    logger.debug("     (This tool will run separate searches for each term split by | and combine results)")
    logger.debug("   - Exclude terms: pandas -zoo -animal")
    logger.debug("   - Channel specific: python coding channel:\"Tech with Tim\"")
    logger.debug("   - Exact title match: intitle:\"complete python course\"")

    logger.debug("\n3. Special Search Types:")
    logger.debug("   - HD videos only: hd hiking in yosemite")
    logger.debug("   - Live streams: live concert")
    logger.debug("   - Long videos (>20 min): long documentary about space")
    logger.debug("   - Short videos (<4 min): short funny cats")
    logger.debug("   - 4K videos: 4k nature documentary")
    logger.debug("   - Creative Commons: creativecommons yoga tutorial")

    logger.debug("\n4. Limitations:")
    logger.debug("   - YouTube search does NOT support regex")
    logger.debug("   - Complex Boolean operators are NOT supported")
    logger.debug("   - Search is limited to ~128 characters")

    logger.debug("\n5. Examples of Effective Queries:")
    logger.debug("   - \"machine learning tutorial\" python beginner -advanced")
    logger.debug("   - earthquake 2025 | tsunami 2025 before:2025-06-01")
    logger.debug("   - intitle:\"how to\" fix iphone screen")
    logger.debug("   - channel:TEDx motivation | inspiration long")

    logger.debug("\n=== END OF GUIDE ===\n")

if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    # Load query and email from .env if present
    env_vars = initialization_result["env_vars"]
    query = env_vars.get("YOUTUBE_QUERY", "")
    recipient_email = env_vars.get("NOTIFICATION_EMAIL", None)
    
    max_videos = 100
    
    # Date filtering options
    use_date_filter = None
    publish_after = None
    publish_before = None

    if use_date_filter == "y":
        date_filter_type = input("\nChoose a date filter option:\n"
                               "1. Last 24 hours\n"
                               "2. Last week\n"
                               "3. Last month\n"
                               "4. Last year\n"
                               "5. Custom date range\n"
                               "Enter choice (1-5): ")
        
        now = datetime.now()
        
        if date_filter_type == '1':  # Last 24 hours
            publish_after = (now - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
            logger.debug(f"Filtering for videos uploaded in the last 24 hours (after {publish_after})")
            
        elif date_filter_type == '2':  # Last week
            publish_after = (now - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ')
            logger.debug(f"Filtering for videos uploaded in the last week (after {publish_after})")

        elif date_filter_type == '3':  # Last month
            publish_after = (now - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
            logger.debug(f"Filtering for videos uploaded in the last month (after {publish_after})")

        elif date_filter_type == '4':  # Last year
            publish_after = (now - timedelta(days=365)).strftime('%Y-%m-%dT%H:%M:%SZ')
            logger.debug(f"Filtering for videos uploaded in the last year (after {publish_after})")
            
        elif date_filter_type == '5':  # Custom date range
            try:
                logger.debug("Enter date in YYYY-MM-DD format")
                start_date = input("Start date (leave blank for no lower bound): ")
                end_date = input("End date (leave blank for no upper bound): ")
                
                if start_date:
                    # Convert to RFC 3339 format with time at 00:00:00
                    publish_after = f"{start_date}T00:00:00Z"
                    logger.debug(f"Lower date bound set to: {publish_after}")
                

                
                if end_date:
                    # Convert to RFC 3339 format with time at 23:59:59
                    publish_before = f"{end_date}T23:59:59Z"
                    logger.debug(f"Upper date bound set to: {publish_before}")

            except Exception as e:
                logger.error(f"Error parsing date: {e}. Proceeding without date filtering.")
                publish_after = None
                publish_before = None
    
    # Location filter options
    use_location_filter = "Y"
    location = None
    location_radius = None
    
    if use_location_filter:

        #elif location_filter_type == '2':
        # print("\nCommon country codes: US (United States), GB (United Kingdom), CA (Canada),")
        # print("                      IN (India), JP (Japan), FR (France), DE (Germany),")
        # print("                      AU (Australia), BR (Brazil), MX (Mexico)")
        region_code = "US"
        if len(region_code) != 2:
                logger.warning("Invalid country code. Using no region filter.")
                region_code = None
        else:
                logger.info(f"Filtering for videos in region: {region_code}")
        #else:
        #    print("Invalid choice. Proceeding without location filter.")
    
    # Language relevance option
    # use_language_filter = input("Do you want to filter by language relevance? (y/n, default: n): ").lower() == 'y'
    # relevance_language = None
    
    # if use_language_filter:
    #     print("\nCommon language codes: en (English), es (Spanish), fr (French), de (German),")
    #     print("                       zh (Chinese), ja (Japanese), ko (Korean), ar (Arabic),")
    #     print("                       hi (Hindi), pt (Portuguese), ru (Russian)")
    #     relevance_language = input("Enter two-letter language code: ").lower()
    #     if len(relevance_language) != 2:
    #         print("Invalid language code. Using no language filter.")
    #         relevance_language = None
    #     else:
    #         print(f"Setting relevance language to: {relevance_language}")

    # Sort order option
    logger.info("\nHow should results be sorted?")
    sort_options = {
        '1': 'relevance',
        '2': 'date',
        '3': 'viewCount',
        '4': 'rating',
        '5': 'title'
    }
    sort_choice = "1"
    
    order = sort_options.get(sort_choice, 'relevance')
    logger.info(f"Sorting results by: {order}")
    
    # Initialize region_code if it's not defined (in case location_filter_type != '2')
    # print("ZZZ",locals())
    # if not region_code in locals():
    #     region_code = None
    
    print("\nStarting YouTube video analysis...")
    analysis_results = analyze_youtube_video_by_title(
        query, 
        recipient_email, 
        max_videos,
        publish_after=publish_after,
        publish_before=publish_before,
        order=order,
        location=location,
        location_radius=location_radius,
        region_code=region_code
        #relevance_language=relevance_language
    )
    
    if analysis_results:
        risk_summaries = [entry for entry in analysis_results if entry['risk_assessment']['risk_level'] in ['high', 'medium']]

        # Get the recipient email from environment variables, falling back to default if not set
        recipient_email = env_vars.get("NOTIFICATION_EMAIL","")
        logger.info(f"Using notification email: {recipient_email}")

        # Call the function to send the notification
        send_consolidated_analysis_notification(recipient_email, risk_summaries)
    
    print(f"\nAnalysis complete. Found and analyzed {len(analysis_results)} videos.")
