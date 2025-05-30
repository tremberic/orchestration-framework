#!/usr/bin/env python3
"""
JWT Generator for Snowflake Authentication

To run this on the command line, enter:
  python3 generateJWT.py --account=<account_identifier> --user=<username> --private_key_file_path=<path_to_private_key_file>
"""

import argparse
import base64
import hashlib
import logging
import sys
from datetime import datetime, timedelta, timezone
from getpass import getpass
from pathlib import Path
from typing import Optional

import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
)
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

# JWT payload field constants
ISSUER = "iss"
EXPIRE_TIME = "exp"
ISSUE_TIME = "iat"
SUBJECT = "sub"

# Default token settings
DEFAULT_LIFETIME_MINUTES = 59
DEFAULT_RENEWAL_DELAY_MINUTES = 54
JWT_ALGORITHM = "RS256"

logger = logging.getLogger(__name__)


def get_private_key_passphrase() -> str:
    """Prompt user for private key passphrase."""
    return getpass("Passphrase for private key: ")


class JWTGeneratorError(Exception):
    """Custom exception for JWT generation errors."""

    pass


class JWTGenerator:
    """
    Creates and signs JWT tokens for Snowflake authentication.

    The generator caches tokens and only regenerates them when they're close to expiration.
    """

    def __init__(
        self,
        account: str,
        user: str,
        private_key_file_path: str,
        lifetime: timedelta = timedelta(minutes=DEFAULT_LIFETIME_MINUTES),
        renewal_delay: timedelta = timedelta(minutes=DEFAULT_RENEWAL_DELAY_MINUTES),
    ):
        """
        Initialize JWT generator.

        Args:
            account: Snowflake account identifier (exclude region info for account locators)
            user: Snowflake username
            private_key_file_path: Path to private key file for signing JWTs
            lifetime: JWT validity duration (default: 59 minutes)
            renewal_delay: Time before renewal (default: 54 minutes)
        """
        logger.info(
            "Creating JWTGenerator - account: %s, user: %s, lifetime: %s, renewal_delay: %s",
            account,
            user,
            lifetime,
            renewal_delay,
        )

        self.account = self._prepare_account_name(account)
        self.user = user.upper()
        self.qualified_username = f"{self.account}.{self.user}"

        self.lifetime = lifetime
        self.renewal_delay = renewal_delay
        self.private_key_file_path = Path(private_key_file_path)
        self.renew_time = datetime.now(timezone.utc)
        self.token: Optional[str] = None

        self.private_key = self._load_private_key()

    def _prepare_account_name(self, raw_account: str) -> str:
        """
        Prepare account identifier for JWT use.

        Removes subdomain, region, and cloud provider information.
        """
        account = raw_account

        if ".global" not in account:
            # Handle general case - remove everything after first dot
            if "." in account:
                account = account.split(".")[0]
        else:
            # Handle replication case - remove everything after first dash
            if "-" in account:
                account = account.split("-")[0]

        return account.upper()

    def _load_private_key(self) -> RSAPrivateKey:
        """Load and return the private key from file."""
        if not self.private_key_file_path.exists():
            raise JWTGeneratorError(
                f"Private key file not found: {self.private_key_file_path}"
            )

        try:
            with open(self.private_key_file_path, "rb") as key_file:
                pem_data = key_file.read()

            # Try loading without passphrase first
            try:
                return load_pem_private_key(pem_data, None, default_backend())
            except TypeError:
                # If that fails, prompt for passphrase
                passphrase = get_private_key_passphrase().encode()
                return load_pem_private_key(pem_data, passphrase, default_backend())

        except Exception as e:
            raise JWTGeneratorError(f"Failed to load private key: {e}") from e

    def _calculate_public_key_fingerprint(self) -> str:
        """Calculate SHA256 fingerprint of the public key."""
        public_key_bytes = self.private_key.public_key().public_bytes(
            Encoding.DER, PublicFormat.SubjectPublicKeyInfo
        )

        sha256_hash = hashlib.sha256(public_key_bytes).digest()
        fingerprint = "SHA256:" + base64.b64encode(sha256_hash).decode("utf-8")

        logger.info("Public key fingerprint: %s", fingerprint)
        return fingerprint

    def get_token(self) -> str:
        """
        Get JWT token, generating a new one if needed.

        Returns cached token unless renewal time has passed.
        """
        now = datetime.now(timezone.utc)

        if self.token is None or self.renew_time <= now:
            logger.info(
                "Generating new token (current time: %s, renewal time: %s)",
                now,
                self.renew_time,
            )

            self.renew_time = now + self.renewal_delay
            public_key_fp = self._calculate_public_key_fingerprint()

            payload = {
                ISSUER: f"{self.qualified_username}.{public_key_fp}",
                SUBJECT: self.qualified_username,
                ISSUE_TIME: now,
                EXPIRE_TIME: now + self.lifetime,
            }

            # Generate the token
            token = jwt.encode(payload, key=self.private_key, algorithm=JWT_ALGORITHM)

            # Handle bytes return type in older PyJWT versions
            if isinstance(token, bytes):
                token = token.decode("utf-8")

            self.token = token

            # Log the decoded payload for verification
            decoded_payload = jwt.decode(
                self.token,
                key=self.private_key.public_key(),
                algorithms=[JWT_ALGORITHM],
            )
            logger.info("Generated JWT with payload: %s", decoded_payload)

        return self.token


def main():
    """Main CLI entry point."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Generate JWT tokens for Snowflake authentication"
    )
    parser.add_argument(
        "--account",
        required=True,
        help='Account identifier (e.g. "myorg-myaccount" for "myorg-myaccount.snowflakecomputing.com")',
    )
    parser.add_argument("--user", required=True, help="Snowflake username")
    parser.add_argument(
        "--private_key_file_path",
        required=True,
        help="Path to private key file for signing JWTs",
    )
    parser.add_argument(
        "--lifetime",
        type=int,
        default=DEFAULT_LIFETIME_MINUTES,
        help=f"JWT validity duration in minutes (default: {DEFAULT_LIFETIME_MINUTES})",
    )
    parser.add_argument(
        "--renewal_delay",
        type=int,
        default=DEFAULT_RENEWAL_DELAY_MINUTES,
        help=f"Minutes before renewal (default: {DEFAULT_RENEWAL_DELAY_MINUTES})",
    )

    args = parser.parse_args()

    try:
        generator = JWTGenerator(
            args.account,
            args.user,
            args.private_key_file_path,
            timedelta(minutes=args.lifetime),
            timedelta(minutes=args.renewal_delay),
        )

        token = generator.get_token()
        print("JWT:")
        print(token)

    except JWTGeneratorError as e:
        logger.error("JWT generation failed: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
