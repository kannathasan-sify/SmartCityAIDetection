# app/services/supabase_client.py
from supabase import create_client, Client
from app.config import settings

supabase: Client = create_client(
    settings.supabase_url,
    settings.supabase_service_key,  # server-side service key — never expose to browser clients
)
