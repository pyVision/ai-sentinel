#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube Social Media Analysis Tool - CLI

This script provides a command-line interface for the YouTube analysis tool.

Usage:
    python youtube_cli.py [--query "search term"] [options]

Options:
    --query TEXT          Search query for YouTube videos (overrides .env setting)
    --email TEXT          Email address for notifications (overrides .env setting)
    --max-videos INTEGER  Maximum number of videos to analyze (default: 3)
    --use-llm             Use LLM for advanced risk analysis (requires litellm)
    --llm-model TEXT      LLM model to use (default: gpt-4o)
    --api-key TEXT        API key for LLM provider
    --no-save             Don't save results to JSON file
    --help                Show this help message

Example:
    python youtube_cli.py --query "crypto scam 2025" --email "analyst@example.com" --max-videos 5

Date: June 26, 2025
"""

import argparse
import sys
import logging
from .youtube_analysis_tool import analyze_youtube_video_by_title, detect_adversarial_risks
from .init_application import initialization_result
from .email_service import send_error_notification, send_consolidated_analysis_notification


# Setup logging
logger = logging.getLogger("youtube_cli")

def parse_args():
    parser = argparse.ArgumentParser(description="YouTube Social Media Analysis Tool")
    parser.add_argument("--query", type=str, help="Search query for YouTube videos (overrides .env setting)")
    parser.add_argument("--email", type=str, help="Email address for notifications (overrides .env setting)")
    parser.add_argument("--max-videos", type=int, default=3, help="Maximum number of videos to analyze")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for advanced risk analysis")
    parser.add_argument("--llm-model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument("--api-key", type=str, help="API key for LLM provider")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to JSON file")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get environment variables
    env_vars = initialization_result["env_vars"]
    
    # Use command line args if provided, otherwise use env vars
    query = args.query if args.query is not None else env_vars.get("YOUTUBE_QUERY", "")
    email = args.email if args.email is not None else env_vars.get("NOTIFICATION_EMAIL", None)
    
    if not query:
        logger.error("No query provided via command line or .env file (YOUTUBE_QUERY)")
        print("Error: No search query provided. Use --query option or set YOUTUBE_QUERY in .env")
        return 1
    
    logger.info("\n=== YouTube Social Media Analysis Tool ===")
    logger.info(f"Query: {query}")
    logger.info(f"Max videos: {args.max_videos}")
    logger.info(f"Email notifications: {'Enabled (' + email + ')' if email else 'Disabled'}")
    logger.info(f"LLM analysis: {'Enabled' if args.use_llm else 'Disabled'}")
    if args.use_llm:
        logger.info(f"LLM model: {args.llm_model}")
    logger.info(f"Save results: {'Disabled' if args.no_save else 'Enabled'}")
    logger.info("="*40)
    
    try:
        # Override the detect_adversarial_risks default parameters if needed
        # if args.use_llm:
        #     # Monkey patch the function temporarily for this run
        #     original_func = detect_adversarial_risks
        #     def patched_detect_adversarial_risks(video_data, comments_df, **kwargs):
        #         return original_func(video_data, comments_df, 
        #                             use_llm=True, 
        #                             llm_model=args.llm_model,
        #                             api_key=args.api_key)
                
        #     # Replace the function
        #     import youtube_analysis_tool
        #     youtube_analysis_tool.detect_adversarial_risks = patched_detect_adversarial_risks
        
        # Run the analysis
        results = analyze_youtube_video_by_title(
            query=query,
            recipient_email=email,
            max_videos=args.max_videos,
            save_results=not args.no_save
        )

        if results:
            risk_summaries = [entry for entry in results if entry['risk_assessment']['risk_level'] in ['high', 'medium']]

            # Get the recipient email from environment variables, falling back to default if not set
            #if not email :
            #recipient_email = env_vars.get("NOTIFICATION_EMAIL", "test@example.com")
            logger.info(f"Using notification email: {email}")

            # Call the function to send the notification
            send_consolidated_analysis_notification(email, risk_summaries)
        
        
        return 0 if results else 1
            
    except KeyboardInterrupt:
        logger.warning("\nAnalysis interrupted by user.")
        return 130
    except Exception as e:
        logger.error(f"\nError: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
