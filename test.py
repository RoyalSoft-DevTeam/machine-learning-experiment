import requests

# Set the token endpoint URL
token_endpoint = 'https://oauth2.googleapis.com/token'

# Set the authorization code, client ID, client secret, and redirect URI
authorization_code = '<authorization_code>'
client_id = '<your_client_id>'
client_secret = '<your_client_secret>'
redirect_uri = '<your_redirect_uri>'

# Prepare the request payload
payload = {
    'code': 'https://accounts.google.com/o/oauth2/auth',
    'client_id': '526231103598-he5qmhvh98skbo188a66hvb680gieb5h.apps.googleusercontent.com',
    'client_secret': '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',
    'redirect_uri': "http://localhost",
    'grant_type': 'authorization_code'
}

# Send the POST request to the token endpoint
response = requests.post(token_endpoint, data=payload)

# Handle the response
if response.status_code == 200:
    # Request successful, parse the JSON response
    token_data = response.json()
    access_token = token_data['access_token']
    refresh_token = token_data['refresh_token']
    # You can now use the access_token to make authorized API requests
else:
    # Request failed, handle the error
    error_message = response.json()['error_description']
    print(f"Token exchange failed: {error_message}")