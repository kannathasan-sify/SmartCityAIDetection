# scripts/seed_db.py
import os
import sys
from pathlib import Path

# Add backend to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.supabase_client import supabase

def seed():
    print("Seeding database...")
    
    cameras = [
        {'id': 'cam_001', 'name': 'Main Junction North', 'lat': 13.0827, 'lng': 80.2707, 'stream_url': '0', 'location': 'Chennai, Anna Salai', 'active': True},
        {'id': 'cam_002', 'name': 'Harbour Express South', 'lat': 13.0569, 'lng': 80.2901, 'stream_url': '0', 'location': 'Chennai, OMR KM 2', 'active': True},
        {'id': 'cam_003', 'name': 'City Central Bridge', 'lat': 13.0732, 'lng': 80.2609, 'stream_url': '0', 'location': 'Chennai, Napier Bridge', 'active': True}
    ]
    
    events = [
        {'camera_id': 'cam_001', 'event_type': 'congestion', 'severity': 'high', 'lat': 13.0827, 'lng': 80.2707, 'description': 'Moderate traffic congestion'},
        {'camera_id': 'cam_002', 'event_type': 'incident', 'severity': 'critical', 'lat': 13.0569, 'lng': 80.2901, 'description': 'Stopped vehicle detected'}
    ]

    try:
        print("Inserting cameras...")
        res_cam = supabase.table('cameras').upsert(cameras).execute()
        print(f"Success: {len(res_cam.data)} cameras seeded.")
        
        print("Inserting events...")
        res_ev = supabase.table('road_events').insert(events).execute()
        print(f"Success: {len(res_ev.data)} events seeded.")
        
        print("Done!")
    except Exception as e:
        print(f"Error seeding: {e}")

if __name__ == "__main__":
    seed()
