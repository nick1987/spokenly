#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_firebase_import():
    """Test if Firebase can be imported and initialized."""
    try:
        print("Testing Firebase import...")
        from firebase_config import initialize_firebase
        print("✅ Firebase config imported successfully")
        
        # Test initialization
        print("Testing Firebase initialization...")
        initialize_firebase()
        print("✅ Firebase initialized successfully")
        
        return True
    except Exception as e:
        print(f"❌ Firebase test failed: {e}")
        return False

def test_environment():
    """Test environment variables."""
    print("\nTesting environment variables...")
    
    required_vars = [
        'DEEPGRAM_API_KEY',
        'FIREBASE_PROJECT_ID'
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value[:10]}..." if len(value) > 10 else f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
    
    # Check for service account file
    service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
    if service_account_path and os.path.exists(service_account_path):
        print(f"✅ Service account file exists: {service_account_path}")
    else:
        print(f"⚠️  Service account file not found: {service_account_path}")
        print("   This is okay for development with default credentials")

def main():
    print("🔥 Firebase Integration Test 🔥\n")
    
    # Test imports
    firebase_ok = test_firebase_import()
    
    # Test environment
    test_environment()
    
    print("\n" + "="*50)
    if firebase_ok:
        print("✅ Firebase integration is ready!")
        print("\nNext steps:")
        print("1. Get your Firebase web app config from Firebase Console")
        print("2. Update src/firebase.js with actual values")
        print("3. Enable Google Sign-In in Firebase Console")
        print("4. Start the backend: python3 main.py")
        print("5. Start the frontend: npm start")
    else:
        print("❌ Firebase integration needs attention")
        print("Check the error messages above and fix any issues")

if __name__ == "__main__":
    main()
