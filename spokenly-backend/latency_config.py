"""
Latency Configuration Profiles for Spokenly
This file contains different configuration profiles optimized for various use cases.
"""

class LatencyProfiles:
    """Different latency profiles for various use cases."""
    
    # Ultra-low latency profile - fastest possible transcription
    ULTRA_LOW = {
        "utterance_end_ms": "500",  # Very short pause detection
        "endpointing": 100,  # Very aggressive endpointing
        "interim_results_interval": 50,  # Very frequent interim results
        "vad_aggressiveness": 1,  # Least aggressive VAD
        "audio_buffer_size": 512,  # Smallest audio buffer
        "chunk_size": 10,  # Smallest chunk size
        "description": "Ultra-low latency - fastest possible transcription with minimal accuracy trade-offs"
    }
    
    # Low latency profile - balanced speed and accuracy
    LOW = {
        "utterance_end_ms": "1000",  # Short pause detection
        "endpointing": 150,  # Aggressive endpointing
        "interim_results_interval": 100,  # Frequent interim results
        "vad_aggressiveness": 1,  # Least aggressive VAD
        "audio_buffer_size": 1024,  # Small audio buffer
        "chunk_size": 20,  # Small chunk size
        "description": "Low latency - good balance between speed and accuracy"
    }
    
    # Standard profile - balanced approach
    STANDARD = {
        "utterance_end_ms": "2000",  # Standard pause detection
        "endpointing": 300,  # Standard endpointing
        "interim_results_interval": 200,  # Standard interim results
        "vad_aggressiveness": 2,  # Standard VAD
        "audio_buffer_size": 2048,  # Standard audio buffer
        "chunk_size": 40,  # Standard chunk size
        "description": "Standard - balanced approach with good accuracy"
    }
    
    # High accuracy profile - slower but more accurate
    HIGH_ACCURACY = {
        "utterance_end_ms": "5000",  # Long pause detection
        "endpointing": 500,  # Conservative endpointing
        "interim_results_interval": 500,  # Less frequent interim results
        "vad_aggressiveness": 3,  # Aggressive VAD
        "audio_buffer_size": 4096,  # Large audio buffer
        "chunk_size": 50,  # Large chunk size
        "description": "High accuracy - slower but more accurate transcription"
    }

def get_profile(profile_name: str = "LOW"):
    """Get a specific latency profile by name."""
    profiles = {
        "ULTRA_LOW": LatencyProfiles.ULTRA_LOW,
        "LOW": LatencyProfiles.LOW,
        "STANDARD": LatencyProfiles.STANDARD,
        "HIGH_ACCURACY": LatencyProfiles.HIGH_ACCURACY
    }
    
    return profiles.get(profile_name.upper(), LatencyProfiles.LOW)

def list_profiles():
    """List all available latency profiles."""
    return {
        "ULTRA_LOW": LatencyProfiles.ULTRA_LOW,
        "LOW": LatencyProfiles.LOW,
        "STANDARD": LatencyProfiles.STANDARD,
        "HIGH_ACCURACY": LatencyProfiles.HIGH_ACCURACY
    }
