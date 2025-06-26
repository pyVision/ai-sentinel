#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Email Service Module

This module provides email notification functionality for the ai-sentinel application.
It supports:
- Regular text and HTML emails
- Error notifications
- Analysis result notifications
- Status notifications

Date: June 23, 2025
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from .init_application import initialization_result

# Default SMTP configuration from environment variables
DEFAULT_SMTP_CONFIG = {
    'server': initialization_result['env_vars'].get('SMTP_SERVER', 'smtp.gmail.com'),
    'port': int(initialization_result['env_vars'].get('SMTP_PORT', 587)),
    'username': initialization_result['env_vars'].get('SMTP_USERNAME', ''),
    'password': initialization_result['env_vars'].get('SMTP_PASSWORD', ''),
    'sender_email': initialization_result['env_vars'].get('FROM_EMAIL', 'ai-sentinel@example.com'),
    'use_tls': bool(initialization_result['env_vars'].get('SMTP_USE_TLS', True)),
}

# Set to True for testing without sending actual emails
TEST_MODE = initialization_result['env_vars'].get('EMAIL_TEST_MODE', 'False').lower() == 'true'


def send_email(
    recipient_email: str,
    subject: str,
    body_text: str = None,
    body_html: str = None,
    smtp_config: Dict[str, Any] = None,
) -> bool:
    """
    Send an email notification
    
    Args:
        recipient_email: Email address to send the notification to
        subject: Email subject line
        body_text: Plain text email body (optional if html_body provided)
        body_html: HTML email body (optional if body_text provided)
        smtp_config: Dictionary with SMTP configuration (optional)
        
    Returns:
        Boolean indicating success or failure
    """
    if not recipient_email:
        print("No recipient email provided. Email not sent.")
        return False
    
    if not body_text and not body_html:
        print("No email content provided. Email not sent.")
        return False
    
    # Use default SMTP configuration if not provided
    if smtp_config is None:
        smtp_config = DEFAULT_SMTP_CONFIG
    
    # Create message
    message = MIMEMultipart("alternative")
    message["From"] = smtp_config.get('sender_email', DEFAULT_SMTP_CONFIG['sender_email'])
    message["To"] = recipient_email
    message["Subject"] = subject
    
    # Add plain text body if provided
    if body_text:
        message.attach(MIMEText(body_text, "plain"))
    
    # Add HTML body if provided (will be preferred by email clients if both are provided)
    if body_html:
        message.attach(MIMEText(body_html, "html"))
    
    # Test mode - don't actually send an email
    if TEST_MODE:
        print("\n==== EMAIL WOULD BE SENT (TEST MODE) ====")
        print(f"To: {recipient_email}")
        print(f"Subject: {subject}")
        if body_text:
            print("\nText Content:")
            print(body_text[:500] + ('...' if len(body_text) > 500 else ''))
        if body_html:
            print("\nHTML Content: [Not displayed in console, but would be included]")
        print("=========================================\n")
        return True
    
    # Connect to SMTP server and send email
    try:
        server = smtplib.SMTP(smtp_config.get('server', DEFAULT_SMTP_CONFIG['server']), 
                             smtp_config.get('port', DEFAULT_SMTP_CONFIG['port']))
        
        if smtp_config.get('use_tls', DEFAULT_SMTP_CONFIG['use_tls']):
            server.starttls()
        
        if smtp_config.get('username') and smtp_config.get('password'):
            server.login(smtp_config.get('username'), smtp_config.get('password'))
        
        server.send_message(message)
        server.quit()
        
        print(f"Email notification sent to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"Failed to send email notification: {e}")
        return False


def send_error_notification(
    recipient_email: str,
    error_info: Dict[str, str],
    smtp_config: Dict[str, Any] = None
) -> bool:
    """
    Send error notification email
    
    Args:
        recipient_email: Email address to send the notification to
        error_info: Dictionary with error information
        smtp_config: Dictionary with SMTP configuration (optional)
        
    Returns:
        Boolean indicating success or failure
    """
    if not recipient_email:
        print("No recipient email provided for error notification.")
        return False
    
    subject = f"ERROR: {error_info.get('operation', 'AI-Sentinel Operation')}"
    
    # Create HTML content with error details
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .container {{ max-width: 600px; margin: 0 auto; }}
            .header {{ background-color: #ff4d4d; padding: 10px; color: white; }}
            .content {{ padding: 15px; }}
            .footer {{ font-size: 12px; color: #666; margin-top: 30px; text-align: center; }}
            .error-box {{ background-color: #ffe6e6; border-left: 4px solid #ff4d4d; padding: 10px; margin: 15px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>AI-Sentinel Error Notification</h2>
            </div>
            
            <div class="content">
                <h3>An error occurred during operation: {error_info.get('operation', 'Unknown Operation')}</h3>
                
                <div class="error-box">
                    <p><strong>Error Message:</strong> {error_info.get('error_message', 'No error message provided')}</p>
                    <p><strong>Timestamp:</strong> {error_info.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
                </div>
                
                <h4>Additional Information:</h4>
                <ul>
    """
    
    # Add any additional information from error_info
    for key, value in error_info.items():
        if key not in ['operation', 'error_message', 'timestamp', 'error_details']:
            html += f"<li><strong>{key}:</strong> {value}</li>"
    
    html += f"""
                </ul>
                
                {'<h4>Error Details:</h4><pre>' + error_info.get('error_details', '') + '</pre>' if error_info.get('error_details') else ''}
                
                <p>Please check the application logs for more information.</p>
            </div>
            
            <div class="footer">
                <p>This is an automated notification from the AI-Sentinel system.</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Create plain text version
    text = f"""
    AI-SENTINEL ERROR NOTIFICATION
    
    An error occurred during operation: {error_info.get('operation', 'Unknown Operation')}
    
    Error Message: {error_info.get('error_message', 'No error message provided')}
    Timestamp: {error_info.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
    
    Additional Information:
    """
    
    for key, value in error_info.items():
        if key not in ['operation', 'error_message', 'timestamp', 'error_details']:
            text += f"- {key}: {value}\n"
    
    if error_info.get('error_details'):
        text += f"\nError Details:\n{error_info.get('error_details')}"
    
    text += "\nPlease check the application logs for more information."
    
    # Send the email
    return send_email(
        recipient_email=recipient_email,
        subject=subject,
        body_text=text,
        body_html=html,
        smtp_config=smtp_config
    )


def send_credential_status_notification(
    recipient_email: str,
    status_info: Dict[str, str],
    smtp_config: Dict[str, Any] = None
) -> bool:
    """
    Send notification about OAuth2 credential status
    
    Args:
        recipient_email: Email address to send the notification to
        status_info: Dictionary with status information
        smtp_config: Dictionary with SMTP configuration (optional)
        
    Returns:
        Boolean indicating success or failure
    """
    if not recipient_email:
        print("No recipient email provided for credential status notification.")
        return False
    
    status = status_info.get('status', 'Unknown')
    is_error = status.lower() in ['failed', 'invalid', 'error']
    
    subject = f"OAuth2 Credential Status: {status}"
    
    # Create HTML content
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .container {{ max-width: 600px; margin: 0 auto; }}
            .header {{ background-color: {('#ff4d4d' if is_error else '#4CAF50')}; 
                       padding: 10px; color: white; }}
            .content {{ padding: 15px; }}
            .status-box {{ background-color: {('#ffe6e6' if is_error else '#e6ffe6')}; 
                          border-left: 4px solid {('#ff4d4d' if is_error else '#4CAF50')}; 
                          padding: 10px; margin: 15px 0; }}
            .footer {{ font-size: 12px; color: #666; margin-top: 30px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>OAuth2 Credential Status Notification</h2>
            </div>
            
            <div class="content">
                <h3>Status: {status}</h3>
                
                <div class="status-box">
                    <p><strong>Message:</strong> {status_info.get('message', 'No message provided')}</p>
                    <p><strong>Timestamp:</strong> {status_info.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
                    <p><strong>Token Path:</strong> {status_info.get('token_path', 'Not specified')}</p>
                </div>
                
                {('<p><strong>Action Required:</strong> Please run the oauth2_credential_manager.py script to create valid credentials.</p>' 
                  if is_error else '<p>No action required. Your credentials are valid.</p>')}
            </div>
            
            <div class="footer">
                <p>This is an automated notification from the AI-Sentinel system.</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Create plain text version
    text = f"""
    OAUTH2 CREDENTIAL STATUS NOTIFICATION
    
    Status: {status}
    
    Message: {status_info.get('message', 'No message provided')}
    Timestamp: {status_info.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
    Token Path: {status_info.get('token_path', 'Not specified')}
    
    {('Action Required: Please run the oauth2_credential_manager.py script to create valid credentials.' 
      if is_error else 'No action required. Your credentials are valid.')}
    """
    
    # Send the email
    return send_email(
        recipient_email=recipient_email,
        subject=subject,
        body_text=text,
        body_html=html,
        smtp_config=smtp_config
    )



def send_consolidated_analysis_notification(
    recipient_email: str,
    risk_summaries: List[Dict[str, Any]],
    smtp_config: Dict[str, Any] = None
) -> bool:
    """
    Send a consolidated email notification with analysis summaries for all high and medium risk videos
    
    Args:
        recipient_email: Email address to send the notification to
        risk_summaries: List of dictionaries with analysis summaries for high/medium risk videos
        smtp_config: Dictionary with SMTP configuration (optional)
        
    Returns:
        Boolean indicating success or failure
    """
    if not recipient_email:
        print("No recipient email provided for consolidated analysis notification.")
        return False
        
    if not risk_summaries:
        print("No risk summaries provided. Email not sent.")
        return False
    
    # Count risks by level
    high_risks = len([s for s in risk_summaries if s.get('risk_assessment', {}).get('risk_level') == 'high'])
    medium_risks = len([s for s in risk_summaries if s.get('risk_assessment', {}).get('risk_level') == 'medium'])
    
    # Set subject based on risk level
    if high_risks > 0:
        subject = f"🚨 {high_risks} HIGH RISK VIDEOS DETECTED - YouTube Analysis Report"
    else:
        subject = f"⚠️ {medium_risks} Medium Risk Videos - YouTube Analysis Report"
    
    # Create HTML content with styling
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .header {{ background-color: #{'#ff4d4d' if high_risks > 0 else '#ffcc00'}; 
                      padding: 15px; color: {'white' if high_risks > 0 else 'black'}; text-align: center; }}
            .summary-box {{ margin-bottom: 30px; border: 1px solid #ddd; border-radius: 5px; overflow: hidden; }}
            .summary-header {{ padding: 10px; color: white; font-weight: bold; }}
            .high-risk {{ background-color: #ff4d4d; }}
            .medium-risk {{ background-color: #ffcc00; color: black; }}
            .summary-content {{ padding: 15px; }}
            .section {{ margin: 15px 0; }}
            .stats {{ display: flex; justify-content: space-between; flex-wrap: wrap; }}
            .stat-item {{ text-align: center; padding: 10px; width: 30%; }}
            .risk-item {{ margin-bottom: 5px; padding: 5px; background-color: #f8f8f8; }}
            .footer {{ font-size: 12px; color: #666; margin-top: 30px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>YouTube Risk Analysis Report</h1>
                <p>{'🚨 ' + str(high_risks) + ' HIGH RISK' if high_risks > 0 else ''} 
                   {' and ' if high_risks > 0 and medium_risks > 0 else ''}
                   {'⚠️ ' + str(medium_risks) + ' MEDIUM RISK' if medium_risks > 0 else ''} videos detected</p>
                <p>Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>This automated analysis has identified {len(risk_summaries)} YouTube videos that present potential risks
                   requiring your attention. Below you'll find a detailed breakdown of each video along with specific risk factors.</p>
            </div>
    """
    
    # Sort summaries - high risks first, then by view count
    sorted_summaries = sorted(
        risk_summaries, 
        key=lambda x: (0 if x.get('risk_assessment', {}).get('risk_level') == 'high' else 1, 
                      -int(x.get('video', {}).get('view_count', 0)))
    )
    
    # Add each risk summary
    for i, summary in enumerate(sorted_summaries, 1):
        risk_level = summary.get('risk_assessment', {}).get('risk_level', 'low')
        
        html += f"""
            <div class="summary-box">
                <div class="summary-header {'high-risk' if risk_level == 'high' else 'medium-risk'}">
                    Video {i}: {risk_level.upper()} RISK
                </div>
                <div class="summary-content">
                    <h3><a href="{summary.get('video', {}).get('url', '#')}">{summary.get('video', {}).get('title', 'Unknown Title')}</a></h3>
                    <p>Channel: {summary.get('video', {}).get('channel', 'Unknown')}</p>
                    
                    <div class="stats">
                        <div class="stat-item">
                            <strong>{summary.get('video', {}).get('view_count', 0):,}</strong>
                            <div>Views</div>
                        </div>
                        <div class="stat-item">
                            <strong>{summary.get('video', {}).get('like_count', 0):,}</strong>
                            <div>Likes</div>
                        </div>
                        <div class="stat-item">
                            <strong>{summary.get('video', {}).get('comment_count', 0):,}</strong>
                            <div>Comments</div>
                        </div>
                    </div>
                    
                    <h4>Risk Factors:</h4>
                    <ul>
                        {''.join([f"<li class='risk-item'>{risk}</li>" for risk in summary.get('risk_assessment', {}).get('summary', ['No specific risks listed'])])}
                    </ul>
                    
                    <h4>Recommendation:</h4>
                    <p>{summary.get('recommendation', 'No recommendation available')}</p>
                </div>
            </div>
        """
    
    # Close HTML
    html += f"""
            <div class="footer">
                <p>This is an automated analysis report from the AI-Sentinel system.</p>
                <p>Please review each video carefully and take appropriate action as needed.</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Create plain text version
    text = f"""
    YOUTUBE RISK ANALYSIS REPORT
    
    {'🚨 ' + str(high_risks) + ' HIGH RISK' if high_risks > 0 else ''}{' and ' if high_risks > 0 and medium_risks > 0 else ''}{'⚠️ ' + str(medium_risks) + ' MEDIUM RISK' if medium_risks > 0 else ''} videos detected
    
    EXECUTIVE SUMMARY:
    This automated analysis has identified {len(risk_summaries)} YouTube videos that present potential risks requiring your attention.
    
    DETAILED RISK ASSESSMENT:
    """
    
    # Add each risk summary to text version
    for i, summary in enumerate(sorted_summaries, 1):
        risk_level = summary.get('risk_assessment', {}).get('risk_level', 'low')
        
        text += f"""
    Video {i}: {risk_level.upper()} RISK
    Title: {summary.get('video', {}).get('title', 'Unknown Title')}
    URL: {summary.get('video', {}).get('url', 'Unknown URL')}
    Channel: {summary.get('video', {}).get('channel', 'Unknown')}
    Stats: Views: {summary.get('video', {}).get('view_count', 0):,}, Likes: {summary.get('video', {}).get('like_count', 0):,}, Comments: {summary.get('video', {}).get('comment_count', 0):,}
    
    Risk Factors:
    """
        
        for risk in summary.get('risk_assessment', {}).get('summary', ['No specific risks listed']):
            text += f"- {risk}\n"
        
        text += f"""
    Recommendation:
    {summary.get('recommendation', 'No recommendation available')}
    
    -------------------
    """
    
    text += f"""
    This is an automated analysis report from the AI-Sentinel system.
    Please review each video carefully and take appropriate action as needed.
    Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
    """
    
    # Send the email
    return send_email(
        recipient_email=recipient_email,
        subject=subject,
        body_text=text,
        body_html=html,
        smtp_config=smtp_config
    )


if __name__ == "__main__":
    # Simple test
    print("Email Service Module - Test Mode")
    print(f"SMTP Config: {DEFAULT_SMTP_CONFIG}")
    print(f"Test Mode: {TEST_MODE}")
    
    # Test sending a simple email
    test_recipient = os.environ.get("TEST_EMAIL_RECIPIENT", "test@example.com")
    send_email(
        recipient_email=test_recipient,
        subject="Test Email from AI-Sentinel",
        body_text="This is a test email from the AI-Sentinel email service module.",
        body_html="<html><body><h1>Test Email</h1><p>This is a <b>test email</b> from the AI-Sentinel email service module.</p></body></html>"
    )
