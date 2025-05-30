"""
Access SPCS services using Snowflake keypair authentication.

This module provides functionality to authenticate with Snowflake using JWT tokens
and connect to SPCS (Snowflake Container Services) endpoints.
"""

import argparse
import json
import logging
import sys
from datetime import timedelta
from typing import Dict, Any, Optional

import requests

from generateJWT import JWTGenerator

# Constants
DEFAULT_JWT_LIFETIME_MINUTES = 59
DEFAULT_JWT_RENEWAL_DELAY_MINUTES = 54
DEFAULT_ENDPOINT_PATH = "/"
OAUTH_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"
HTTP_SUCCESS = 200

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the application."""
    args = _parse_args()

    try:
        token = _get_token(args)
        snowflake_jwt = _token_exchange(
            token,
            endpoint=args.endpoint,
            role=args.role,
            snowflake_account_url=args.snowflake_account_url,
            snowflake_account=args.account,
        )
        spcs_url = f"https://{args.endpoint}{args.endpoint_path}"
        _connect_to_spcs(snowflake_jwt, spcs_url, args.payload)
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


def _get_token(args: argparse.Namespace) -> str:
    """
    Generate a JWT token using the provided credentials.

    Args:
        args: Parsed command line arguments containing authentication details

    Returns:
        str: Generated JWT token

    Raises:
        Exception: If token generation fails
    """
    try:
        token_generator = JWTGenerator(
            args.account,
            args.user,
            args.private_key_file_path,
            timedelta(minutes=args.lifetime),
            timedelta(minutes=args.renewal_delay),
        )
        return token_generator.get_token()
    except Exception as e:
        logger.error(f"Failed to generate JWT token: {e}")
        raise


def _token_exchange(
    token: str,
    role: Optional[str],
    endpoint: str,
    snowflake_account_url: Optional[str],
    snowflake_account: str,
) -> str:
    """
    Exchange JWT token for Snowflake OAuth token.

    Args:
        token: JWT token to exchange
        role: Optional role to assume
        endpoint: Target endpoint for scope
        snowflake_account_url: Optional custom Snowflake account URL
        snowflake_account: Snowflake account identifier

    Returns:
        str: Snowflake OAuth token response

    Raises:
        requests.RequestException: If token exchange fails
        AssertionError: If response status is not 200
    """
    scope_role = f"session:role:{role}" if role else None
    scope = f"{scope_role} {endpoint}" if scope_role else endpoint

    data = {
        "grant_type": OAUTH_GRANT_TYPE,
        "scope": scope,
        "assertion": token,
    }

    # Construct OAuth URL
    if snowflake_account_url:
        oauth_url = f"{snowflake_account_url}/oauth/token"
    else:
        oauth_url = f"https://{snowflake_account}.snowflakecomputing.com/oauth/token"

    logger.info(f"OAuth URL: {oauth_url}")

    try:
        response = requests.post(oauth_url, data=data, timeout=30)
        response.raise_for_status()
        logger.info("Successfully obtained Snowflake OAuth token")
        return response.text
    except requests.RequestException as e:
        logger.error(f"Failed to exchange token: {e}")
        raise


def _connect_to_spcs(token: str, url: str, payload: Optional[str] = None) -> None:
    """
    Connect to SPCS endpoint with the provided token.

    Args:
        token: Snowflake OAuth token
        url: SPCS endpoint URL
        payload: Optional JSON payload string to send

    Raises:
        requests.RequestException: If connection fails
    """
    headers = {
        "Authorization": f'Snowflake Token="{token}"',
        "Content-Type": "application/json",
    }

    request_payload = _parse_payload(payload)

    try:
        if request_payload is not None:
            response = requests.post(
                url, headers=headers, json=request_payload, timeout=30
            )
        else:
            response = requests.post(url, headers=headers, timeout=30)

        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response: {response.text}")

        if response.status_code != HTTP_SUCCESS:
            logger.warning(f"Non-success status code: {response.status_code}")

    except requests.RequestException as e:
        logger.error(f"Failed to connect to SPCS: {e}")
        raise


def _parse_payload(payload: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Parse the provided payload string.

    Args:
        payload: Optional JSON payload string

    Returns:
        Optional[Dict[str, Any]]: Parsed payload dictionary or None if no payload provided
    """
    if payload is None:
        logger.info("No payload provided")
        return None

    try:
        parsed_payload = json.loads(payload)
        logger.info("Successfully parsed custom payload")
        return parsed_payload
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON payload: {e}")
        raise ValueError(f"Invalid JSON payload: {e}")


def _parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Access SPCS services using Snowflake keypair authentication"
    )

    # Required arguments
    parser.add_argument(
        "--account",
        required=True,
        help='The account identifier (e.g., "myorganization-myaccount" for '
        '"myorganization-myaccount.snowflakecomputing.com")',
    )
    parser.add_argument(
        "--user",
        required=True,
        help="The user name for authentication",
    )
    parser.add_argument(
        "--private_key_file_path",
        required=True,
        help="Path to the private key file used for signing the JWT",
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="The ingress endpoint of the service",
    )

    # Optional arguments
    parser.add_argument(
        "--lifetime",
        type=int,
        default=DEFAULT_JWT_LIFETIME_MINUTES,
        help=f"JWT validity period in minutes (default: {DEFAULT_JWT_LIFETIME_MINUTES})",
    )
    parser.add_argument(
        "--renewal_delay",
        type=int,
        default=DEFAULT_JWT_RENEWAL_DELAY_MINUTES,
        help=f"JWT renewal delay in minutes (default: {DEFAULT_JWT_RENEWAL_DELAY_MINUTES})",
    )
    parser.add_argument(
        "--role",
        help="The role to use for the session. If not provided, uses the default role",
    )
    parser.add_argument(
        "--endpoint-path",
        default=DEFAULT_ENDPOINT_PATH,
        help=f"The URL path for the ingress endpoint (default: {DEFAULT_ENDPOINT_PATH})",
    )
    parser.add_argument(
        "--snowflake_account_url",
        help="Custom Snowflake account URL (e.g., https://myorganization-myaccount.snowflakecomputing.com)",
    )
    parser.add_argument(
        "--payload",
        help="JSON payload to send in the request.",
    )

    return parser.parse_args()


def _setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


if __name__ == "__main__":
    main()
