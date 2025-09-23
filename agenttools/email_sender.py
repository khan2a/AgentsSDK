from __future__ import annotations

import base64
import os.path
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying scopes, delete token.json before running again
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

# Set paths relative to the script location
token_path = 'token.json'
credentials_path = 'credentials.json'


def gmail_authenticate():
    creds = None
    # token.json stores user access/refresh tokens after first login
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    # If no valid credentials, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Check if credentials file exists
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"Credentials file not found at: {credentials_path}\n"
                    f"Please ensure you have downloaded the OAuth 2.0 Client ID "
                    f"JSON file from Google Cloud Console and saved it as 'credentials.json' "
                    f"in the tools directory.",
                )

            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), SCOPES,
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for next run
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)


def create_message(sender, to, subject, message_text):
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes())
    return {'raw': raw.decode()}


def send_message(service, user_id, message):
    return service.users().messages().send(userId=user_id, body=message).execute()


def send_email(sender, to, subject, body):
    service = gmail_authenticate()
    message = create_message(sender, to, subject, body)
    return send_message(service, sender, message)


def send_html_email(sender, to, subject, html_body):
    service = gmail_authenticate()
    message = MIMEText(html_body, 'html')
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes())
    body = {'raw': raw.decode()}
    return send_message(service, sender, body)


if __name__ == '__main__':
    print(send_email('me', 'engineer.atique@gmail.com', 'Hello from Gmail API', 'This is a test email sent using Gmail API with Python.'))
    print('Email sent successfully!')
