import socket
import time

UDP_PORT = 4444

def main():
    print(f"--- UDP Port {UDP_PORT} Sniffer ---")
    print("This will show if ANY data is reaching this computer.")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", UDP_PORT))
        print(f"Successfully bound to port {UDP_PORT}. Listening...")
    except Exception as e:
        print(f"Error binding to port {UDP_PORT}: {e}")
        return

    sock.settimeout(5.0)
    count = 0
    start_t = time.time()
    
    try:
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                count += 1
                print(f"[{count}] Received {len(data)} bytes from {addr}")
            except socket.timeout:
                print("... Still waiting for data (None received in last 5 seconds) ...")
                
    except KeyboardInterrupt:
        print("\nStopping sniffer.")

if __name__ == "__main__":
    main()
