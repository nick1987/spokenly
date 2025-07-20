import React, { useRef, useState } from 'react';

const App = () => {
  const [transcript, setTranscript] = useState('');
  const socketRef = useRef(null);
  const mediaRecorderRef = useRef(null);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    socketRef.current = new WebSocket('ws://localhost:8000/ws');

    socketRef.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.transcript) setTranscript((prev) => prev + message.transcript + ' ');
    };

    mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });

    mediaRecorderRef.current.ondataavailable = (event) => {
      if (event.data.size > 0 && socketRef.current.readyState === WebSocket.OPEN) {
        socketRef.current.send(event.data);
      }
    };

    mediaRecorderRef.current.start(250);
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Spokenly – Live Transcriber</h1>
      <button onClick={startRecording}>Start Transcription</button>
      <p>{transcript}</p>
    </div>
  );
};

export default App;
