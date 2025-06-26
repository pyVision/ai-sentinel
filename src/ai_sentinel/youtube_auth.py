#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube API Authentication Module

This module provides authentication functionality for the YouTube API,
separating credential management from the application logic.

Date: June 23, 2025
"""

import os
from datetime import datetime
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle

# Import the credential manager and email service
from .oauth2_credential_manager import OAuth2CredentialManager
from .email_service import send_error_notification, send_credential_status_notification

# API configuration
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'
DEFAULT_TOKEN_PATH = os.getenv('YOUTUBE_TOKEN_PATH', 'token.pickle')


def get_authenticated_service(recipient_email=None):
    """
    Get an authenticated YouTube API client using saved credentials,
    with email notification on authentication failure.
    
    Args:
        recipient_email: Optional email for error notifications
        
    Returns:
        tuple: (youtube_service, error_info)
            - youtube_service: Authenticated service or None if authentication failed
            - error_info: Dict with error details if authentication failed, None otherwise
    """
    # Create credential manager instance
    manager = OAuth2CredentialManager(token_path=DEFAULT_TOKEN_PATH)
    
    # Check if credentials are valid
    is_valid, message, credentials = manager.validate_credentials(test_api_call=False)
    
    if not is_valid:
        error_info = {
            'operation': 'YouTube API Authentication',
            'video_id': 'N/A',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error_message': f"Authentication error: {message}",
            'youtube_link': 'https://www.youtube.com/'
        }
        
        # Send notification if email provided
        if recipient_email:
            status_info = {
                'status': 'Failed',
                'message': f"Authentication error: {message}. Please run the oauth2_credential_manager.py script to create new credentials.",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'token_path': DEFAULT_TOKEN_PATH
            }
            send_credential_status_notification(recipient_email, status_info)
            
        print(f"Authentication failed: {message}")
        print("Please run the oauth2_credential_manager.py script to create new credentials.")
        return None, error_info
    
    try:
        # Refresh token if needed
        if credentials.expired:
            credentials.refresh(Request())
            # Save refreshed credentials
            with open(DEFAULT_TOKEN_PATH, 'wb') as token:
                pickle.dump(credentials, token)
            print("Refreshed credentials and updated token file")
        
        # Build the YouTube service
        print("Building YouTube API client...")
        youtube = build(API_SERVICE_NAME, API_VERSION, credentials=credentials)
        return youtube, None
        
    except Exception as e:
        error_info = {
            'operation': 'YouTube API Authentication',
            'video_id': 'N/A',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error_message': f"Error building YouTube client: {str(e)}",
            'youtube_link': 'https://www.youtube.com/'
        }
        
        # Send notification if email provided
        if recipient_email:
            send_error_notification(recipient_email, error_info)
        
        print(f"Error building YouTube client: {e}")
        return None, error_info