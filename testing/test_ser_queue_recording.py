#!/usr/bin/env python3
"""
Test Script: Local Recording + SER Queue

This script records audio locally from the microphone and sends it to the SER service
queue for processing. Records continuously in 10-second chunks.

Usage:
    python test_ser_queue_recording.py [OPTIONS]
    
Options:
    --user-id UUID       User UUID (default: test user UUID)
    --url URL            SER service URL (default: http://localhost:8000)
    --duration SECONDS   Recording duration per chunk in seconds (default: 10)
    --max-chunks N       Maximum number of chunks to send (default: infinite)
"""

import os
import sys
import argparse
import logging
import tempfile
import wave
import time
import requests
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# SER Service URL (can be overridden via --url argument or SER_SERVICE_URL env var)
SER_SERVICE_URL = os.getenv("SER_SERVICE_URL", "http://localhost:8000")
SER_ENDPOINT = "/ser/analyze-speech"
SER_TIMEOUT = 30

# Default test user ID
DEFAULT_USER_ID = "8517c97f-66ef-4955-86ed-531013d33d3e"


def capture_audio_from_mic(duration_seconds: float, sample_rate: int = 16000, chunk_size: int = 1600) -> Optional[list]:
    """
    Capture audio from microphone for specified duration.
    
    Args:
        duration_seconds: How long to record
        sample_rate: Sample rate in Hz (default: 16000)
        chunk_size: Frames per chunk (default: 1600)
    
    Returns:
        List of audio chunks (bytes) if successful, None if failed
    """
    try:
        import pyaudio
    except ImportError:
        logger.error("=" * 60)
        logger.error("ERROR: pyaudio is not installed!")
        logger.error("=" * 60)
        logger.error("To install pyaudio:")
        logger.error("  Windows: pip install pyaudio")
        logger.error("  Linux:   sudo apt-get install portaudio19-dev && pip install pyaudio")
        logger.error("  Mac:     brew install portaudio && pip install pyaudio")
        logger.error("=" * 60)
        return None
    
    pa = None
    stream = None
    
    try:
        logger.info(f"Initializing microphone (rate: {sample_rate}Hz, chunk: {chunk_size})...")
        pa = pyaudio.PyAudio()
        
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        logger.info("Microphone active - speak now!")
        logger.info(f"Recording for {duration_seconds} seconds...")
        
        # Calculate number of chunks needed
        chunks_per_second = sample_rate // chunk_size
        total_chunks = int(duration_seconds * chunks_per_second)
        
        audio_chunks = []
        for i in range(total_chunks):
            try:
                chunk = stream.read(chunk_size, exception_on_overflow=False)
                audio_chunks.append(chunk)
                if (i + 1) % chunks_per_second == 0:
                    elapsed = (i + 1) // chunks_per_second
                    logger.info(f"Recording... {elapsed}/{int(duration_seconds)} seconds")
            except Exception as e:
                logger.error(f"Error capturing audio chunk: {e}")
                break
        
        logger.info(f"Captured {len(audio_chunks)} audio chunks")
        return audio_chunks
        
    except Exception as e:
        logger.error(f"Error during audio capture: {e}", exc_info=True)
        return None
    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                logger.debug(f"Error closing stream: {e}")
        if pa:
            try:
                pa.terminate()
            except Exception as e:
                logger.debug(f"Error terminating PyAudio: {e}")


def save_audio_to_wav(audio_chunks: list, sample_rate: int, output_path: Path) -> bool:
    """
    Save audio chunks to WAV file.
    
    Args:
        audio_chunks: List of audio chunks (bytes)
        sample_rate: Sample rate in Hz
        output_path: Path to save WAV file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Write all chunks
            for chunk in audio_chunks:
                wav_file.writeframes(chunk)
        
        logger.info(f"Audio saved to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving audio to WAV: {e}", exc_info=True)
        return False


def send_audio_to_queue(audio_file_path: Path, service_url: str, user_id: str) -> Optional[dict]:
    """
    Send audio file to SER service queue.
    
    Args:
        audio_file_path: Path to WAV audio file
        service_url: Base URL of SER service
        user_id: UUID of the user
    
    Returns:
        Dictionary with queue status if successful, None if failed
    """
    try:
        url = f"{service_url}{SER_ENDPOINT}"
        logger.info(f"Sending audio to SER queue: {url}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Audio file: {audio_file_path}")
        
        if not audio_file_path.exists():
            logger.error(f"Audio file not found: {audio_file_path}")
            return None
        
        with open(audio_file_path, 'rb') as audio_file:
            files = {'file': (audio_file_path.name, audio_file, 'audio/wav')}
            data = {'user_id': user_id}
            response = requests.post(url, files=files, data=data, timeout=SER_TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            queue_size = result.get("queue_size", 0)
            logger.info(f"✅ Audio chunk queued successfully (queue size: {queue_size})")
            return result
        else:
            logger.error(f"❌ SER service error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error(f"❌ Request timeout after {SER_TIMEOUT} seconds")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Connection error - is the SER service running at {service_url}?")
        logger.error(f"   Error: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Error sending audio to SER service: {e}", exc_info=True)
        return None


def test_continuous_recording(duration_seconds: float, user_id: str, max_chunks: Optional[int] = None):
    """
    Test SER queue with continuous microphone recording.
    
    Args:
        duration_seconds: Recording duration per chunk
        user_id: User UUID
        max_chunks: Maximum number of chunks to send (None = infinite)
    """
    logger.info("=" * 60)
    logger.info("TEST: Continuous Recording + SER Queue")
    logger.info("=" * 60)
    logger.info(f"Recording duration per chunk: {duration_seconds} seconds")
    logger.info(f"User ID: {user_id}")
    logger.info(f"SER Service: {SER_SERVICE_URL}")
    logger.info(f"Dashboard: {SER_SERVICE_URL}/ser/dashboard")
    logger.info(f"Max chunks: {max_chunks or 'infinite'}")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop recording")
    logger.info("=" * 60)
    
    chunk_count = 0
    
    try:
        while True:
            chunk_count += 1
            logger.info("")
            logger.info(f"[Chunk #{chunk_count}] Starting recording...")
            
            # Step 1: Capture audio
            sample_rate = 16000
            chunk_size = 1600
            audio_chunks = capture_audio_from_mic(duration_seconds, sample_rate, chunk_size)
            
            if not audio_chunks:
                logger.error("Failed to capture audio")
                logger.warning("Continuing to next chunk...")
                time.sleep(1)
                continue
            
            # Step 2: Save to temporary WAV file
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', prefix=f'ser_test_chunk{chunk_count}_')
            temp_path = Path(temp_audio_file.name)
            temp_audio_file.close()
            
            if not save_audio_to_wav(audio_chunks, sample_rate, temp_path):
                logger.error("Failed to save audio file")
                logger.warning("Continuing to next chunk...")
                time.sleep(1)
                continue
            
            try:
                # Step 3: Send to SER queue
                logger.info(f"[Chunk #{chunk_count}] Sending to SER queue...")
                result = send_audio_to_queue(temp_path, SER_SERVICE_URL, user_id)
                
                if result:
                    queue_size = result.get("queue_size", 0)
                    logger.info(f"[Chunk #{chunk_count}] ✅ Queued (queue size: {queue_size})")
                    logger.info(f"   View queue at: {SER_SERVICE_URL}/ser/dashboard")
                else:
                    logger.error(f"[Chunk #{chunk_count}] ❌ Failed to queue")
                    
            finally:
                # Clean up temp file
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                        logger.debug(f"Cleaned up temp file: {temp_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file: {e}")
            
            # Check max_chunks limit
            if max_chunks and chunk_count >= max_chunks:
                logger.info("")
                logger.info(f"Reached maximum chunks ({max_chunks}). Stopping.")
                break
            
            # Small delay before next chunk
            logger.info(f"[Chunk #{chunk_count}] Waiting 1 second before next recording...")
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Recording interrupted by user (Ctrl+C)")
        logger.info(f"Total chunks sent: {chunk_count}")
        logger.info(f"View queue at: {SER_SERVICE_URL}/ser/dashboard")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise


def main():
    """Main test function."""
    global SER_SERVICE_URL
    
    parser = argparse.ArgumentParser(
        description="Test SER queue with continuous microphone recording",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record 10-second chunks continuously (infinite loop, press Ctrl+C to stop)
  python test_ser_queue_recording.py
  
  # Record 5-second chunks, send 10 chunks then stop
  python test_ser_queue_recording.py --duration 5 --max-chunks 10
  
  # Specify user ID and service URL
  python test_ser_queue_recording.py --url http://localhost:8000 --user-id "your-uuid-here"
        """
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration per chunk in seconds (default: 10.0)"
    )
    
    parser.add_argument(
        "--user-id",
        type=str,
        default=DEFAULT_USER_ID,
        help=f"User UUID (default: {DEFAULT_USER_ID})"
    )
    
    parser.add_argument(
        "--url",
        type=str,
        help=f"SER service URL (default: {SER_SERVICE_URL} or from SER_SERVICE_URL env var)"
    )
    
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to send (default: infinite)"
    )
    
    args = parser.parse_args()
    
    # Override service URL if provided
    if args.url:
        SER_SERVICE_URL = args.url.rstrip('/')
    
    # Validate user ID format (basic UUID check)
    try:
        import uuid
        uuid.UUID(args.user_id)
    except ValueError:
        logger.error(f"Invalid user_id format: {args.user_id}. Must be a valid UUID.")
        sys.exit(1)
    
    # Run test
    test_continuous_recording(
        args.duration, 
        args.user_id,
        max_chunks=args.max_chunks
    )


if __name__ == "__main__":
    main()

