#!/usr/bin/env node

console.log(`
🔥 Firebase Setup Helper for Spokenly 🔥

To complete the Firebase setup, you need to:

1. Go to your Firebase Console: https://console.firebase.google.com/
2. Select your project: "spokenly-e132e"
3. Click the gear icon (⚙️) next to "Project Overview"
4. Select "Project settings"
5. Scroll down to "Your apps" section
6. If you don't see a web app, click the web icon (</>) to add one
7. Copy the configuration object that looks like this:

const firebaseConfig = {
  apiKey: "AIzaSy...",
  authDomain: "spokenly-e132e.firebaseapp.com",
  projectId: "spokenly-e132e",
  storageBucket: "spokenly-e132e.appspot.com",
  messagingSenderId: "123456789012",
  appId: "1:123456789012:web:abcdef1234567890"
};

8. Replace the placeholder values in src/firebase.js with your actual values

Also, make sure to:
- Enable Google Sign-In in Authentication > Sign-in method
- Add "localhost" to authorized domains for development
- Download the service account key for the backend

Need help? Check the FIREBASE_SETUP.md file for detailed instructions.
`);
