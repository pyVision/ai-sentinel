#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google OAuth2 Credential Manager

This script manages OAuth2 credentials for Google APIs, providing functionality to:
- Create new credentials through OAuth2 flow
- Force create new credentials (overwriting existing ones)
- Delete existing credentials
- Validate existing credentials

Date: June 23, 2025
"""

import os
import argparse
import sys
import pickle
from datetime import datetime
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Import the email service
from .email_service import send_credential_status_notification

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import googleapiclient.errors

from .init_application import initialization_result
# Default configuration, now loaded from env if available
env_vars = initialization_result["env_vars"]
DEFAULT_TOKEN_PATH = env_vars.get("GOOGLE_TOKEN_PATH", "token.pickle")
DEFAULT_CLIENT_SECRETS_PATH = env_vars.get("GOOGLE_CLIENT_SECRETS_PATH", "client_secret.json")
DEFAULT_SCOPES = [
    'https://www.googleapis.com/auth/youtube.readonly', 
    'https://www.googleapis.com/auth/youtube.force-ssl',
    'https://www.googleapis.com/auth/youtube',
    'https://www.googleapis.com/auth/youtubepartner'
]

class OAuth2CredentialManager:
    """Manages Google OAuth2 credentials"""
    
    def __init__(self, client_secrets_file=DEFAULT_CLIENT_SECRETS_PATH, 
                 token_path=DEFAULT_TOKEN_PATH, 
                 scopes=DEFAULT_SCOPES):
        """
        Initialize the credential manager
        
        Args:
            client_secrets_file: Path to the client secrets JSON file
            token_path: Path where credentials will be stored
            scopes: List of OAuth scopes to request
        """
        self.client_secrets_file = client_secrets_file
        self.token_path = token_path
        self.scopes = scopes
    
    def create_credentials(self, force=False):
        """
        Create OAuth2 credentials via user authorization flow
        
        Args:
            force: If True, create new credentials even if existing ones are present
            
        Returns:
            tuple: (success_flag, message)
        """
        # Check if credentials already exist and force is False
        if not force and os.path.exists(self.token_path):
            return False, f"Credentials already exist at {self.token_path}. Use --force to override."
        
        # Check if client secrets file exists
        if not os.path.exists(self.client_secrets_file):
            return False, f"Client secrets file not found at {self.client_secrets_file}"
        
        try:
            # Start the OAuth flow
            print(f"Starting OAuth flow for scopes: {', '.join(self.scopes)}")
            flow = InstalledAppFlow.from_client_secrets_file(
                self.client_secrets_file, 
                self.scopes
            )
            
            # Run the local server flow
            credentials = flow.run_local_server(port=8080)
            
            # Save credentials to file
            with open(self.token_path, 'wb') as token_file:
                pickle.dump(credentials, token_file)
                
            return True, f"Successfully created and stored credentials at {self.token_path}"
            
        except Exception as e:
            return False, f"Failed to create credentials: {str(e)}"
    
    def delete_credentials(self):
        """
        Delete existing credentials if they exist
        
        Returns:
            tuple: (success_flag, message)
        """
        if not os.path.exists(self.token_path):
            return False, f"No credentials found at {self.token_path}"
        
        try:
            os.remove(self.token_path)
            return True, f"Successfully deleted credentials at {self.token_path}"
        except Exception as e:
            return False, f"Failed to delete credentials: {str(e)}"
    
    def validate_credentials(self, test_api_call=True):
        """
        Check if stored credentials exist and are valid
        
        Args:
            test_api_call: If True, attempt a test API call to verify credentials
            
        Returns:
            tuple: (is_valid, message, credentials_object)
        """
        credentials = None
        
        # Check if file exists
        if not os.path.exists(self.token_path):
            return False, "No credentials file found", None
        
        try:
            # Load credentials
            with open(self.token_path, 'rb') as token_file:
                credentials = pickle.load(token_file)
            
            # Check if expired
            if credentials.expired and not credentials.refresh_token:
                return False, "Credentials expired and no refresh token available", credentials
                
            # Try to refresh if expired
            if credentials.expired:
                credentials.refresh(Request())
                
                # Save refreshed credentials
                with open(self.token_path, 'wb') as token_file:
                    pickle.dump(credentials, token_file)
                
                print("Refreshed and saved updated credentials")
            
            # Make a test API call if requested
            if test_api_call:
                try:
                    youtube = build('youtube', 'v3', credentials=credentials)
                    # Just get the channels.list response for the authenticated user
                    channels_response = youtube.channels().list(
                        part="snippet",
                        mine=True
                    ).execute()
                    
                    channel_name = channels_response['items'][0]['snippet']['title']
                    return True, f"Credentials valid and active (authenticated as: {channel_name})", credentials
                    
                except googleapiclient.errors.HttpError as e:
                    return False, f"Credentials failed test API call: {str(e)}", credentials
            
            return True, "Credentials appear valid", credentials
            
        except Exception as e:
            return False, f"Error validating credentials: {str(e)}", None
    
    # Using send_credential_status_notification from email_service module instead
    # of implementing email functionality here


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Google OAuth2 Credential Manager")
    
    # Action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--create', action='store_true', help="Create new credentials")
    action_group.add_argument('--force-create', action='store_true', help="Force create new credentials (overwrite existing)")
    action_group.add_argument('--delete', action='store_true', help="Delete existing credentials")
    action_group.add_argument('--validate', action='store_true', help="Validate existing credentials")
    
    # Optional arguments
    parser.add_argument('--secrets', type=str, default=DEFAULT_CLIENT_SECRETS_PATH, 
                        help=f"Path to client secrets file (default: {DEFAULT_CLIENT_SECRETS_PATH})")
    parser.add_argument('--token', type=str, default=DEFAULT_TOKEN_PATH,
                        help=f"Path to save/load token file (default: {DEFAULT_TOKEN_PATH})")
    parser.add_argument('--email', type=str, help="Email address for status notifications")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create credential manager instance
    manager = OAuth2CredentialManager(
        client_secrets_file=args.secrets,
        token_path=args.token
    )
    
    status_info = {
        'status': 'Unknown',
        'message': '',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'token_path': args.token
    }
    
    # Perform the requested action
    if args.create:
        success, message = manager.create_credentials(force=False)
        status_info['status'] = 'Success' if success else 'Failed'
        status_info['message'] = message
        
    elif args.force_create:
        success, message = manager.create_credentials(force=True)
        status_info['status'] = 'Success' if success else 'Failed'
        status_info['message'] = message
        
    elif args.delete:
        success, message = manager.delete_credentials()
        status_info['status'] = 'Success' if success else 'Failed'
        status_info['message'] = message
        
    elif args.validate:
        is_valid, message, _ = manager.validate_credentials()
        status_info['status'] = 'Valid' if is_valid else 'Invalid'
        status_info['message'] = message
    
    # Print result to console
    print(f"\nStatus: {status_info['status']}")
    print(f"Message: {status_info['message']}")
    
    # Send email notification if requested
    if args.email:
        send_credential_status_notification(args.email, status_info)
    
    # Return success/failure code
    return 0 if status_info['status'] in ['Success', 'Valid'] else 1


if __name__ == "__main__":
    sys.exit(main())