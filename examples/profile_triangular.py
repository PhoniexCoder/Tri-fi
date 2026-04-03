import socket, time, json
from csi_frame_parser import CSIFrameParser, CSIFrame, VitalsPacket

UDP_PORT = 4444
parser = CSIFrameParser()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
sock.settimeout(1.0)

print("="*50)
print("  TRIANGULAR SIGNAL PROFILER (10 Seconds)")
print("="*50)

stats = {i: {"rssi": [], "motion": []} for i in [1,2,3]}
start = time.time()

try:
    while time.time() - start < 10:
        try:
            data, _ = sock.recvfrom(4096)
            res = parser.parse(data)
            if isinstance(res, VitalsPacket):
                if res.node_id in [1,2,3]:
                    stats[res.node_id]["motion"].append(res.motion_energy)
            elif isinstance(res, CSIFrame):
                if res.node_id in [1,2,3]:
                    stats[res.node_id]["rssi"].append(res.rssi)
        except socket.timeout: continue

    print("\n--- RESULTS FOR YOUR ROOM ---")
    for i in [1,2,3]:
        r = stats[i]["rssi"]
        m = stats[i]["motion"]
        if r:
            print(f"Node {i}: RSSI Avg={sum(r)/len(r):.1f}, Motion Avg={sum(m)/len(m):.3f} (samples: {len(r)})")
        else:
            print(f"Node {i}: NO DATA RECEIVED")
    print("="*50)

except KeyboardInterrupt:
    pass
finally:
    sock.close()
