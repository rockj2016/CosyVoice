import subprocess
import time
import requests
import os
import signal
import sys

def verify_api():
    # Start the API server
    print("Starting API server...")
    process = subprocess.Popen(
        ['python', 'api.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(10)  # Give it some time to load models
    
    api_key = os.getenv('AUTODL_API_KEY', 'autodl-tts-secret-key-2024')
    url = "http://127.0.0.1:8000/tts"
    
    try:
        # Test 1: Valid Request
        print("\nTest 1: Valid Request")
        payload = {"text": "这是一个测试。"}
        headers = {"X-AutoDL-API-Key": api_key}
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200 and response.headers['content-type'] == 'audio/wav':
            print("PASS: Valid request returned 200 and audio/wav")
            with open("test_output.wav", "wb") as f:
                f.write(response.content)
            print("Saved test_output.wav")
        else:
            print(f"FAIL: Valid request returned {response.status_code}")
            print(response.text)

        # Test 2: Invalid API Key
        print("\nTest 2: Invalid API Key")
        headers_invalid = {"X-AutoDL-API-Key": "wrong-key"}
        response = requests.post(url, json=payload, headers=headers_invalid)
        
        if response.status_code == 403:
            print("PASS: Invalid key returned 403")
        else:
            print(f"FAIL: Invalid key returned {response.status_code}")
            
        # Test 3: Long Text
        print("\nTest 3: Long Text")
        long_text = "测" * 101
        payload_long = {"text": long_text}
        response = requests.post(url, json=payload_long, headers=headers)
        
        if response.status_code == 422:
            print("PASS: Long text returned 422")
        else:
            print(f"FAIL: Long text returned {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nStopping API server...")
        # Start by trying to terminate gently
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
        # Print server output for debugging
        stdout, stderr = process.communicate()
        if stdout:
            print(f"Server STDOUT:\n{stdout}")
        if stderr:
             print(f"Server STDERR:\n{stderr}")

if __name__ == "__main__":
    verify_api()
