import React from 'react';
import { AuthProvider } from './contexts/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import TranscriptionApp from './components/TranscriptionApp';
import './App.css';

const App = () => {
  return (
    <AuthProvider>
      <ProtectedRoute>
        <TranscriptionApp />
      </ProtectedRoute>
    </AuthProvider>
  );
};

export default App;
