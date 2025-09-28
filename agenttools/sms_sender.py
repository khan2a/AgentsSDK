from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from vonage import Auth
from vonage import HttpClientOptions
from vonage import Vonage
from vonage_sms import SmsMessage
from vonage_sms import SmsResponse

load_dotenv(override=True)
logger = logging.getLogger(__name__)


def send_sms(
        to_number: str | None = os.getenv('SMS_RECIPIENT_NUMBER'),
        from_number: str | None = os.getenv('VONAGE_LVN'),
        message_text: str | None = 'Hello from Vonage SMS API with Python!',
) -> SmsResponse:
    """Send an SMS using Vonage API."""
    # Create an Auth instance
    auth = Auth(api_key=os.getenv('VONAGE_API_KEY'), api_secret=os.getenv('VONAGE_API_SECRET'))
    # Create HttpClientOptions instance
    # (not required unless you want to change options from the defaults)
    options = HttpClientOptions(api_host='api.nexmo.com', timeout=30)

    # Create a Vonage instance
    vonage = Vonage(auth=auth, http_client_options=options)
    message = SmsMessage(to=to_number, from_=from_number, text=message_text)
    response = vonage.sms.send(message)
    print(response.model_dump_json(exclude_unset=True))
    return response


if __name__ == '__main__':
    response = send_sms(
        message_text='Yes it works!',
    )
    if response.messages[0].status == '0':
        print('Message sent successfully.')
    else:
        print(f"Message failed with error: {response.messages[0]['error-text']}")
