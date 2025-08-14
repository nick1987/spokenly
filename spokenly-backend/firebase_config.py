import os
import logging
from typing import Optional, Dict, Any
from firebase_admin import credentials, auth, initialize_app
from firebase_admin.auth import UserRecord
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK
def initialize_firebase():
    """Initialize Firebase Admin SDK with service account or default credentials."""
    try:
        # Check if Firebase is already initialized
        try:
            # Try to get the default app
            from firebase_admin import _apps
            if _apps:
                logger.info("Firebase already initialized")
                return
        except:
            pass
        
        # Try to use service account file
        service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
        
        # If not set, try to find the service account file in common locations
        if not service_account_path:
            possible_paths = [
                "spokenlyServiceAccountKey.json",
                "../spokenlyServiceAccountKey.json",
                "../../spokenlyServiceAccountKey.json",
                os.path.join(os.path.dirname(__file__), "..", "spokenlyServiceAccountKey.json"),
                os.path.join(os.path.dirname(__file__), "..", "..", "spokenlyServiceAccountKey.json")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    service_account_path = path
                    logger.info(f"Found service account file at: {path}")
                    break
        
        if service_account_path and os.path.exists(service_account_path):
            cred = credentials.Certificate(service_account_path)
            initialize_app(cred)
            logger.info(f"Firebase initialized with service account: {service_account_path}")
        else:
            # Use default credentials (for Google Cloud deployment)
            initialize_app()
            logger.info("Firebase initialized with default credentials")
            
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        raise

# Security scheme for JWT tokens
security = HTTPBearer()

async def verify_firebase_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserRecord:
    """Verify Firebase ID token and return user record."""
    try:
        token = credentials.credentials
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token['uid']
        user_record = auth.get_user(user_id)
        return user_record
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(user: UserRecord = Depends(verify_firebase_token)) -> UserRecord:
    """Get current authenticated user."""
    return user

def create_custom_token(uid: str) -> str:
    """Create a custom token for a user (if needed for advanced use cases)."""
    try:
        return auth.create_custom_token(uid)
    except Exception as e:
        logger.error(f"Failed to create custom token: {e}")
        raise

def get_user_by_email(email: str) -> Optional[UserRecord]:
    """Get user by email address."""
    try:
        return auth.get_user_by_email(email)
    except:
        return None

def create_user(email: str, display_name: str = None) -> UserRecord:
    """Create a new user in Firebase Auth."""
    try:
        user_properties = {
            'email': email,
            'email_verified': True,
        }
        if display_name:
            user_properties['display_name'] = display_name
            
        return auth.create_user(**user_properties)
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise

# Initialize Firebase when module is imported
initialize_firebase()
