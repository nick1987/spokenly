# Firebase Setup Guide for Spokenly

## Step 1: Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Create a project" or "Add project"
3. Enter a project name (e.g., "spokenly-app")
4. Choose whether to enable Google Analytics (optional)
5. Click "Create project"

## Step 2: Enable Authentication

1. In your Firebase project, go to "Authentication" in the left sidebar
2. Click "Get started"
3. Go to the "Sign-in method" tab
4. Click on "Google" provider
5. Enable it and configure:
   - Project support email: your email
   - Authorized domains: localhost (for development)
6. Click "Save"

## Step 3: Get Firebase Configuration

1. In your Firebase project, click the gear icon (⚙️) next to "Project Overview"
2. Select "Project settings"
3. Scroll down to "Your apps" section
4. Click the web icon (</>)
5. Register your app with a nickname (e.g., "spokenly-web")
6. Copy the Firebase configuration object

## Step 4: Update Frontend Configuration

Replace the placeholder configuration in `src/firebase.js`:

```javascript
const firebaseConfig = {
  apiKey: "your-actual-api-key",
  authDomain: "your-project-id.firebaseapp.com",
  projectId: "your-project-id",
  storageBucket: "your-project-id.appspot.com",
  messagingSenderId: "your-messaging-sender-id",
  appId: "your-app-id"
};
```

## Step 5: Get Service Account Key (Backend)

1. In Firebase project settings, go to "Service accounts" tab
2. Click "Generate new private key"
3. Download the JSON file
4. Place it in your backend directory (e.g., `spokenly-backend/serviceAccountKey.json`)
5. Update your `.env` file:

```env
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_SERVICE_ACCOUNT_PATH=./serviceAccountKey.json
```

## Step 6: Install Dependencies

### Backend
```bash
cd spokenly-backend
pip install -r requirements.txt
```

### Frontend
```bash
cd spokenly-frontend
npm install
```

## Step 7: Run the Application

### Backend
```bash
cd spokenly-backend
python main.py
```

### Frontend
```bash
cd spokenly-frontend
npm start
```

## Security Notes

1. **Never commit your service account key** to version control
2. Add `serviceAccountKey.json` to your `.gitignore` file
3. For production, use environment variables for sensitive data
4. Configure authorized domains in Firebase for your production domain

## Troubleshooting

- **CORS errors**: Make sure your backend CORS settings allow your frontend domain
- **Authentication errors**: Verify your Firebase configuration is correct
- **WebSocket errors**: Check that your backend is running and accessible
