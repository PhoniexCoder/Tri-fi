import asyncio
import json
import websockets

async def poll():
    try:
        async with websockets.connect("ws://localhost:3000/ws/sensing") as ws:
            # Let it warm up for 1 second
            await asyncio.sleep(1)
            msg = await ws.recv()
            data = json.loads(msg)
            nodes = data.get("nodes", [])
            print("--- HARDWARE STATUS ---")
            print(f"Nodes Connected: {len(nodes)}")
            for n in nodes:
                print(f" - ESP32 Node ID: {n.get('node_id', 'Unknown')} | RSSI: {n.get('rssi', -100)} dBm")
    except Exception as e:
        print(f"Server Offline or Error: {e}")

asyncio.run(poll())
