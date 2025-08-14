import { initializeApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider } from 'firebase/auth';
import { getAnalytics } from 'firebase/analytics';

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCv-0suu1saf40OS4C3z_WuUR2exLbFNBI",
  authDomain: "spokenly-e132e.firebaseapp.com",
  projectId: "spokenly-e132e",
  storageBucket: "spokenly-e132e.firebasestorage.app",
  messagingSenderId: "187424591539",
  appId: "1:187424591539:web:965c7cf328c528b9766ac6",
  measurementId: "G-SGN8GW7M53"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
// eslint-disable-next-line no-unused-vars
const analytics = getAnalytics(app);

// Initialize Firebase Authentication and get a reference to the service
export const auth = getAuth(app);

// Google Auth Provider
export const googleProvider = new GoogleAuthProvider();
googleProvider.addScope('email');
googleProvider.addScope('profile');

export default app;
