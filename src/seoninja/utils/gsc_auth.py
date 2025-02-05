"""Google Search Console authentication utilities."""
import os
from typing import Dict, Any
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']
REDIRECT_URI = 'http://localhost:8080'

def get_gsc_credentials(client_secrets_file: str = 'client_secrets.json') -> Dict[str, Any]:
    """Get Google Search Console credentials.
    
    Args:
        client_secrets_file: Path to client secrets file from Google Cloud Console
        
    Returns:
        Dict containing credentials info that can be passed to KeywordResearch
    """
    creds = None
    # Token file stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
            
    # If there are no valid credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, SCOPES)
            # Use specific redirect URI without trailing slash
            creds = flow.run_local_server(
                port=8080,
                redirect_uri_trailing_slash=False
            )
            
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
            
    return {
        'token': creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': creds.scopes
    } 