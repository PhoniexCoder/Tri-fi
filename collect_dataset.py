import socket
import sys
import time
import os
import argparse
import numpy as np

# We import your exact parser to ensure no mismatches
sys.path.append(os.path.join(os.path.dirname(__file__), 'examples'))
from csi_frame_parser import CSIFrameParser, CSIFrame

def main():
    parser = argparse.ArgumentParser(description="Collect raw WiFi CSI telemetry for custom Neural Network training.")
    parser.add_argument("--label", type=int, required=True, 
                        help="0 = EMPTY ROOM. 1 = HUMAN PRESENT (in any zone, breathing, or moving).")
    parser.add_argument("--time", type=int, default=60, help="Recording time in seconds per run (Default: 60)")
    args = parser.parse_args()

    os.makedirs("custom_dataset", exist_ok=True)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 4444))
    
    csi_parser = CSIFrameParser()
    
    print("="*50)
    print(f"📡 DATA COLLECTION SCRIPT")
    print(f"Target Nodes: 1, 2, and 3 (Simultaneous Recording)")
    print(f"Class Label: {args.label} ({'EMPTY room / Outside' if args.label == 0 else 'HUMAN PRESENT'})")
    print(f"Duration:    {args.time} seconds")
    print("="*50)
    print("\nTake your position... Recording starts in 3 seconds!")
    time.sleep(3)
    print("🔴 RECORDING...")

    start_t = time.time()
    frames = {1: [], 2: [], 3: []}

    # Listen loop
    while time.time() - start_t < args.time:
        try:
            data, _ = sock.recvfrom(4096)
        except socket.timeout:
            continue
        
        res = csi_parser.parse(data)
        if isinstance(res, CSIFrame) and res.node_id in frames:
            frames[res.node_id].append(res.amplitude[:64])

    print(f"\n✅ Done! Captured frames:")
    for nid, arr in frames.items():
        print(f" - Node {nid}: {len(arr)} frames")
    
    for nid, arr in frames.items():
        if len(arr) >= 100:
            out_file = f"custom_dataset/node{nid}_label{args.label}_{int(time.time())}.npy"
            np.save(out_file, np.array(arr))
            print(f"💾 Saved Node {nid} to {out_file}")

if __name__ == "__main__":
    main()
