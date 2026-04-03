import socket

def run_bridge():
    print("[UDP Bridge] Binding to Windows external port 5005...")
    sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_in.bind(("0.0.0.0", 5005))
    
    sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print("[UDP Bridge] Successfully intercepting ESP32 packets!")
    print("[UDP Bridge] Forwarding heavily to WSL Docker VM at 172.29.225.131:5005...")
    
    packet_count = 0
    while True:
        data, addr = sock_in.recvfrom(65535)
        packet_count += 1
        sock_out.sendto(data, ("172.29.225.131", 5005))
        
        if packet_count % 500 == 0:
            print(f" -> Bridged {packet_count} raw CSI packets from {addr} to Docker!")

if __name__ == "__main__":
    run_bridge()
