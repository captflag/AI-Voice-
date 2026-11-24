"""The front desk: a page staff keep open to take transfers from the bot.

    python staff.py        then open http://localhost:7870 and start a shift

Needs the same LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET as the bot. The page joins the desk room,
plays the bot's spoken briefing, shows it as text, and on Accept moves into the caller's room with the
microphone on.
"""
import os
import re
from pathlib import Path

from aiohttp import web

from transfer import DESK, token


async def page(request):
    return web.FileResponse(Path(__file__).with_name("staff.html"))


async def desk_token(request):
    name = request.query.get("name", "").strip()[:40] or "Staff"
    identity = "staff-" + (re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "member")
    return web.json_response({"url": os.environ["LIVEKIT_URL"], "token": token(DESK, identity, name)})


app = web.Application()
app.add_routes([web.get("/", page), web.get("/token", desk_token)])

if __name__ == "__main__":
    web.run_app(app, host="localhost", port=7870)
