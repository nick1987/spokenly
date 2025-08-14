# Spokenly Firebase Authentication Integration

## Overview

This integration adds Google Sign-In authentication to your Spokenly speech-to-text application using Firebase Authentication. Users can now sign in with their Google accounts and have their transcription sessions tracked per user.

## Features Added

### 🔐 Authentication
- **Google Sign-In**: One-click authentication with Google accounts
- **Protected Routes**: Unauthenticated users are redirected to login
- **User Sessions**: Each user's transcription sessions are tracked separately
- **Token Management**: Automatic token refresh and validation

### 🎯 User Experience
- **Modern UI**: Clean, responsive design with user profile display
- **Session Management**: Users can see their session history
- **Real-time Updates**: Live transcription with user-specific sessions
- **Error Handling**: Comprehensive error messages and validation

### 🔧 Technical Features
- **WebSocket Authentication**: Secure WebSocket connections with Firebase tokens
- **Backend Integration**: FastAPI endpoints for user management
- **Session Tracking**: User-specific session storage and retrieval
- **CORS Support**: Proper cross-origin resource sharing configuration

## Architecture

```
Frontend (React)          Backend (FastAPI)         Firebase
     |                         |                        |
     |-- Google Sign-In ------>|                        |
     |<-- Firebase Token ------|                        |
     |                         |                        |
     |-- WebSocket + Token --->|-- Verify Token ------>|
     |<-- Live Transcript -----|                        |
     |                         |                        |
     |-- Get User Sessions --->|                        |
     |<-- Session List --------|                        |
```

## File Structure

```
spokenly/
├── spokenly-backend/
│   ├── firebase_config.py          # Firebase Admin SDK setup
│   ├── main.py                     # Updated with auth endpoints
│   ├── requirements.txt            # Added Firebase dependencies
│   └── .env                        # Firebase configuration
├── spokenly-frontend/
│   ├── src/
│   │   ├── firebase.js             # Firebase client config
│   │   ├── contexts/
│   │   │   └── AuthContext.js      # Authentication context
│   │   ├── components/
│   │   │   ├── Login.js            # Google sign-in component
│   │   │   ├── Login.css           # Login styles
│   │   │   ├── ProtectedRoute.js   # Route protection
│   │   │   ├── TranscriptionApp.js # Enhanced transcription UI
│   │   │   └── TranscriptionApp.css # Transcription styles
│   │   └── App.jsx                 # Updated main app
│   ├── package.json                # Added Firebase dependencies
│   └── FIREBASE_SETUP.md           # Setup instructions
└── FIREBASE_INTEGRATION_README.md  # This file
```

## API Endpoints

### Authentication Endpoints
- `GET /auth/me` - Get current user information
- `GET /auth/verify` - Verify authentication token

### User Management
- `GET /user/sessions` - Get all sessions for authenticated user

### WebSocket
- `WS /ws/{session_id}?token={firebase_token}` - Authenticated WebSocket connection

## Setup Instructions

1. **Follow the Firebase Setup Guide**: See `spokenly-frontend/FIREBASE_SETUP.md`
2. **Install Dependencies**: Run `npm install` and `pip install -r requirements.txt`
3. **Configure Environment**: Update Firebase config in both frontend and backend
4. **Start the Application**: Run both backend and frontend servers

## Usage

### For Users
1. Visit the application
2. Click "Sign in with Google"
3. Grant microphone permissions
4. Start transcribing speech
5. View session history and manage transcripts

### For Developers
1. **Adding New Protected Routes**: Wrap components with `<ProtectedRoute>`
2. **Accessing User Data**: Use `useAuth()` hook in components
3. **Making Authenticated Requests**: Include Firebase token in headers
4. **WebSocket Authentication**: Pass token as query parameter

## Security Considerations

- ✅ Firebase tokens are verified on the backend
- ✅ WebSocket connections require valid authentication
- ✅ User sessions are isolated per user
- ✅ Service account keys are excluded from version control
- ✅ CORS is properly configured for security

## Performance Optimizations

- **Token Caching**: Firebase tokens are cached and refreshed automatically
- **Minimal Latency**: Authentication only happens during initial connection
- **Efficient Sessions**: User sessions are tracked in memory for speed
- **Optimized UI**: React components are optimized for performance

## Troubleshooting

### Common Issues

1. **"Firebase not initialized"**
   - Check Firebase configuration in `src/firebase.js`
   - Verify project settings in Firebase Console

2. **"Authentication failed"**
   - Ensure Google Sign-In is enabled in Firebase Console
   - Check authorized domains include your development URL

3. **"WebSocket connection failed"**
   - Verify backend is running on correct port
   - Check Firebase token is being passed correctly

4. **"CORS errors"**
   - Ensure backend CORS settings allow your frontend domain
   - Check Firebase authorized domains

### Debug Mode

Enable debug logging by setting:
```javascript
// In firebase.js
const app = initializeApp(firebaseConfig);
console.log('Firebase initialized:', app);
```

## Future Enhancements

- [ ] User preferences and settings
- [ ] Session export functionality
- [ ] Multi-language support per user
- [ ] Real-time collaboration features
- [ ] Advanced analytics and insights
- [ ] Mobile app support

## Support

For issues or questions:
1. Check the Firebase Setup Guide
2. Review the troubleshooting section
3. Check Firebase Console for configuration issues
4. Verify all dependencies are installed correctly
