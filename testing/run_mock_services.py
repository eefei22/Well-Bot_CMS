#!/usr/bin/env python3
"""
Helper script to start mock model services for fusion service testing.

This script starts all three mock services (SER, FER, Vitals) in separate processes.

Usage:
    python testing/run_mock_services.py
    
    Press Ctrl+C to stop all services.
"""

import sys
import os
import subprocess
import signal
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Service configurations
SERVICES = {
    "ser": {"port": 8005, "modality": "speech"},
    "fer": {"port": 8006, "modality": "face"},
    "vitals": {"port": 8007, "modality": "vitals"}
}

# Store process references
processes = []


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nShutting down all services...")
    for proc in processes:
        try:
            proc.terminate()
        except:
            pass
    
    # Wait for processes to terminate
    for proc in processes:
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()
    
    print("All services stopped.")
    sys.exit(0)


def start_service(service_name: str, port: int, modality: str):
    """Start a single mock service."""
    script_path = os.path.join(os.path.dirname(__file__), "mock_model_services.py")
    
    print(f"Starting {service_name.upper()} service on port {port}...")
    
    proc = subprocess.Popen(
        [sys.executable, script_path, service_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    processes.append(proc)
    return proc


def main():
    """Main entry point."""
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 80)
    print("MOCK MODEL SERVICES LAUNCHER")
    print("=" * 80)
    print("\nStarting mock services:")
    print("  SER:    http://localhost:8005")
    print("  FER:    http://localhost:8006")
    print("  Vitals: http://localhost:8007")
    print("\nPress Ctrl+C to stop all services\n")
    print("=" * 80)
    
    # Start all services
    for service_name, config in SERVICES.items():
        start_service(service_name, config["port"], config["modality"])
        time.sleep(0.5)  # Small delay between starts
    
    print("\nAll services started!")
    print("Services are running in background.")
    print("Check logs above for service status.")
    print("\nTo test, run: python testing/test_fusion_e2e.py")
    print("\nPress Ctrl+C to stop all services...\n")
    
    # Keep script running
    try:
        while True:
            # Check if any process has died
            for i, proc in enumerate(processes):
                if proc.poll() is not None:
                    service_name = list(SERVICES.keys())[i]
                    print(f"\n⚠ {service_name.upper()} service stopped unexpectedly (exit code: {proc.returncode})")
                    # Optionally restart it
                    # proc = start_service(service_name, SERVICES[service_name]["port"], SERVICES[service_name]["modality"])
            
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()



