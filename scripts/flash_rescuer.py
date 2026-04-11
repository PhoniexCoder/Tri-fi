#!/usr/bin/env python3
"""
Trifi Rescuer Flashing Script
-----------------------------
Provides a one-command flash for the ESP8266 Rescuer device,
similar to the ESP32-S3 provisioning workflow.

Requirements:
    1. arduino-cli (https://arduino.github.io/arduino-cli/latest/)
    2. ESP8266 core installed in Arduino

Usage:
    python scripts/flash_rescuer.py --port COM8 --ssid "MyWiFi" --password "secret" --target-ip 192.168.1.100
"""

import argparse
import os
import subprocess
import sys
import tempfile
import shutil

def find_arduino_cli():
    """Find the arduino-cli executable."""
    # 1. Check if it's in the PATH
    cli_path = shutil.which("arduino-cli")
    if cli_path:
        return cli_path
    
    # 2. Check if it's in the current scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, "arduino-cli.exe")
    if os.path.exists(local_path):
        return local_path
        
    return None

def setup_esp8266_core(cli_cmd):
    """Ensure ESP8266 core is installed in arduino-cli."""
    print("Checking for ESP8266 platform...")
    
    # 1. Add the ESP8266 Board Manager URL
    esp8266_url = "https://arduino.esp8266.com/stable/package_esp8266com_index.json"
    try:
        subprocess.check_call([
            cli_cmd, "config", "add", "board_manager.additional_urls", esp8266_url
        ])
    except:
        pass # Might already be added or config is read-only
    
    # 2. Update Index
    print("Updating index...")
    subprocess.check_call([cli_cmd, "core", "update-index"])
    
    # 3. Check if installed
    output = subprocess.check_output([cli_cmd, "core", "list"]).decode("utf-8")
    if "esp8266:esp8266" not in output:
        print("Installing ESP8266 core (this may take a minute)...")
        subprocess.check_call([cli_cmd, "core", "install", "esp8266:esp8266"])
        print("ESP8266 core installed!")
    else:
        print("ESP8266 core is already installed.")

def main():
    parser = argparse.ArgumentParser(description="Flash the Rescuer ESP8266 with a single command")
    parser.add_argument("--port", required=True, help="Serial port (e.g. COM8, /dev/ttyUSB0)")
    parser.add_argument("--ssid", required=True, help="WiFi SSID")
    parser.add_argument("--password", required=True, help="WiFi Password")
    parser.add_argument("--target-ip", required=True, help="Backend Server IP Address")
    parser.add_argument("--fqbn", default="esp8266:esp8266:nodemcuv2", help="Fully Qualified Board Name (default: nodemcuv2)")
    
    args = parser.parse_args()

    ino_path = os.path.join("firmware", "rescuer_esp8266.ino")
    if not os.path.exists(ino_path):
        print(f"Error: {ino_path} not found.")
        sys.exit(1)

    print(f"--- Preparing Rescuer Firmware ---")
    print(f"Target: {args.target_ip} ({args.ssid})")

    # Read the template
    with open(ino_path, "r") as f:
        content = f.read()

    # Replace placeholders
    content = content.replace('const char* ssid     = "YOUR_SSID";', f'const char* ssid     = "{args.ssid}";')
    content = content.replace('const char* password = "YOUR_PASSWORD";', f'const char* password = "{args.password}";')
    content = content.replace('const char* backend_ip = "192.168.1.100";', f'const char* backend_ip = "{args.target_ip}";')

    # Create temporary .ino file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_ino_dir = os.path.join(tmpdir, "rescuer_esp8266")
        os.makedirs(tmp_ino_dir)
        tmp_ino_path = os.path.join(tmp_ino_dir, "rescuer_esp8266.ino")
        
        with open(tmp_ino_path, "w") as f:
            f.write(content)

        print(f"--- Compiling and Uploading ---")
        cli_cmd = find_arduino_cli()
        if not cli_cmd:
            print("\nError: 'arduino-cli' not found in PATH or 'scripts/' folder.")
            sys.exit(1)

        # Ensure environment is ready
        try:
            setup_esp8266_core(cli_cmd)
        except Exception as e:
            print(f"Warning: Automatic core setup failed: {e}")
            print("Trying to proceed anyway...")

        try:
            # 1. Compile
            print(f"Running: {cli_cmd} compile...")
            subprocess.check_call([
                cli_cmd, "compile",
                "--fqbn", args.fqbn,
                tmp_ino_dir
            ])

            # 2. Upload
            print(f"Running: {cli_cmd} upload to {args.port}...")
            subprocess.check_call([
                cli_cmd, "upload",
                "-p", args.port,
                "--fqbn", args.fqbn,
                tmp_ino_dir
            ])

            print("\nSUCCESS: Rescuer device flashed and configured!")
        except subprocess.CalledProcessError as e:
            print(f"\nError: Command failed. Ensure 'arduino-cli' is installed and {args.fqbn} core is available.")
            sys.exit(1)
        except FileNotFoundError:
            print("\nError: 'arduino-cli' not found. Please install it to use one-command flashing.")
            sys.exit(1)

if __name__ == "__main__":
    main()
