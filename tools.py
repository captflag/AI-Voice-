"""Phase 5: what the receptionist can do. An appointment book in SQLite, and the tools Claude calls.

The top half is plain Python with no voice or LLM in it, so it can be tested on its own:

    python tools.py --check

The bottom half wraps those functions as Pipecat tools. Claude sees each tool's name, its
docstring and its arguments; the docstrings are therefore instructions to the model.
"""
import os
import sqlite3
import sys
import tempfile
from contextlib import closing
from datetime import datetime, time, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "appointments.db")
if os.environ.get("APPOINTMENTS_DB") == "fresh":
    # Evals set this so every bot process gets an empty book of its own and never touches the real one.
    DB_PATH = os.path.join(tempfile.mkdtemp(prefix="appointments-"), "appointments.db")
OPEN, CLOSE = time(9), time(17)   # Monday to Friday
SLOT = timedelta(minutes=30)


class Refused(Exception):
    """A request the clinic cannot do. The message is written to be spoken to the caller."""


def connect(path=DB_PATH):
    db = sqlite3.connect(path)
    # start is the primary key: one chair, so the database itself refuses a double booking.
    db.execute("CREATE TABLE IF NOT EXISTS appointments "
               "(start TEXT PRIMARY KEY, name TEXT NOT NULL, phone TEXT NOT NULL, service TEXT NOT NULL)")
    return db


def say(when):
    """A time the way a person says it, because this text ends up in the bot's mouth."""
    return f"{when:%A} {when.day} {when:%B} at {when.hour % 12 or 12}:{when:%M} {when:%p}"


def parse(date, clock):
    """The model sends dates and times as text. Anything malformed is refused, not guessed."""
    try:
        return datetime.fromisoformat(f"{date.strip()}T{clock.strip()}")
    except ValueError:
        raise Refused(f"I could not read {date!r} {clock!r}. Dates must be YYYY-MM-DD and times HH:MM in 24-hour form.")


def digits(phone):
    number = "".join(ch for ch in phone if ch.isdigit())
    if len(number) < 7:
        raise Refused("That phone number looks too short. Ask the caller to say it again, digit by digit.")
    return number


def check_slot(when, now):
    if when.weekday() > 4:
        raise Refused("The clinic is closed at weekends. It is open Monday to Friday.")
    if not OPEN <= when.time() < CLOSE:
        raise Refused("That is outside opening hours, which are 9 AM to 5 PM.")
    if when.minute % 30 or when.second:
        raise Refused("Appointments start on the hour or the half hour.")
    if when <= now:
        raise Refused("That time has already passed.")


def free_slots(db, day, now):
    """Every bookable start time on one day that is still free."""
    booked = {row[0] for row in db.execute("SELECT start FROM appointments WHERE start LIKE ?", (f"{day:%Y-%m-%d}%",))}
    slots, when = [], datetime.combine(day, OPEN)
    while when.time() < CLOSE and day.weekday() <= 4:
        if when > now and when.isoformat() not in booked:
            slots.append(when)
        when += SLOT
    return slots


def book(db, when, name, phone, service, now):
    check_slot(when, now)
    try:
        with db:
            db.execute("INSERT INTO appointments VALUES (?, ?, ?, ?)", (when.isoformat(), name.strip(), digits(phone), service.strip()))
    except sqlite3.IntegrityError:
        raise Refused(f"{say(when)} has just been taken.")
    return when


def reschedule(db, phone, when, now):
    """Move the caller's next appointment, found by phone number, to a new time."""
    check_slot(when, now)
    row = db.execute("SELECT start FROM appointments WHERE phone = ? AND start > ? ORDER BY start LIMIT 1",
                     (digits(phone), now.isoformat())).fetchone()
    if not row:
        raise Refused("I cannot find an upcoming appointment under that phone number.")
    try:
        with db:
            db.execute("UPDATE appointments SET start = ? WHERE start = ?", (when.isoformat(), row[0]))
    except sqlite3.IntegrityError:
        raise Refused(f"{say(when)} is already taken.")
    return datetime.fromisoformat(row[0]), when


# ---- the tools Claude can call -------------------------------------------------------------
# ponytail: these run SQLite on the event loop. That is fine for a local file that answers in
# microseconds; a real calendar API over the network would need to be awaited instead.

from pipecat.frames.frames import EndWorkerFrame, FunctionCallResultProperties, TTSSpeakFrame


async def check_availability(params, date: str):
    """List the free appointment times on one day. Call this before offering any time.

    Args:
        date: The day to check, as YYYY-MM-DD.
    """
    try:
        day = parse(date, "00:00").date()
        with closing(connect()) as db:
            slots = free_slots(db, day, datetime.now())
        if not slots:
            await params.result_callback({"day": f"{day:%A} {day.day} {day:%B}", "free_times": [], "note": "Nothing is free that day."})
        else:
            times = [f"{s.hour % 12 or 12}:{s:%M} {s:%p}" for s in slots]
            await params.result_callback({"day": f"{day:%A} {day.day} {day:%B}", "free_times": times})
    except Refused as problem:
        await params.result_callback({"error": str(problem)})


async def book_appointment(params, date: str, time: str, name: str, phone: str, service: str):
    """Book an appointment. Only call this after the caller has confirmed their name and phone number.

    Args:
        date: The day, as YYYY-MM-DD.
        time: The start time in 24-hour HH:MM, on the hour or half hour.
        name: The caller's full name.
        phone: The caller's phone number, digits only.
        service: What the visit is for, such as cleaning, check-up or filling.
    """
    try:
        with closing(connect()) as db:
            when = book(db, parse(date, time), name, phone, service, datetime.now())
        await params.result_callback({"booked": say(when), "name": name, "service": service})
    except Refused as problem:
        await params.result_callback({"error": str(problem)})


async def reschedule_appointment(params, phone: str, new_date: str, new_time: str):
    """Move the caller's existing appointment to a new time. It is looked up by phone number.

    Args:
        phone: The phone number the appointment was booked under, digits only.
        new_date: The new day, as YYYY-MM-DD.
        new_time: The new start time in 24-hour HH:MM, on the hour or half hour.
    """
    try:
        with closing(connect()) as db:
            old, new = reschedule(db, phone, parse(new_date, new_time), datetime.now())
        await params.result_callback({"moved_from": say(old), "moved_to": say(new)})
    except Refused as problem:
        await params.result_callback({"error": str(problem)})


async def end_call(params, farewell: str):
    """Say goodbye and hang up. Call this once, when the caller has nothing more to ask.

    Args:
        farewell: The one short goodbye sentence to say to the caller before hanging up.
    """
    # The goodbye is an argument because, left to say it first, the model sometimes hung up in silence.
    await params.result_callback({"ended": True}, properties=FunctionCallResultProperties(run_llm=False))
    await params.llm.push_frame(TTSSpeakFrame(farewell))
    await params.llm.push_frame(EndWorkerFrame())  # queued behind the goodbye, so the call ends after it plays


TOOLS = [check_availability, book_appointment, reschedule_appointment, end_call]


def check():
    now = datetime(2026, 11, 23, 10, 0)  # a Monday morning
    tuesday = datetime(2026, 11, 24)
    with closing(connect(":memory:")) as db:
        assert len(free_slots(db, tuesday.date(), now)) == 16                       # 9:00 to 4:30
        assert free_slots(db, now.date(), now)[0] == datetime(2026, 11, 23, 10, 30)  # today: only what is still ahead
        assert free_slots(db, datetime(2026, 11, 28).date(), now) == []             # Saturday

        book(db, tuesday.replace(hour=9, minute=30), "Sam Lee", "741 333 333", "cleaning", now)
        assert tuesday.replace(hour=9, minute=30) not in free_slots(db, tuesday.date(), now)

        for when, reason in ((tuesday.replace(hour=9, minute=30), "taken"), (tuesday.replace(hour=18), "opening hours"),
                             (tuesday.replace(hour=9, minute=10), "half hour"), (datetime(2026, 11, 28, 10), "weekends"),
                             (datetime(2026, 11, 23, 9), "passed")):
            try:
                book(db, when, "Ana Roy", "5550001111", "check-up", now)
                raise AssertionError(f"should have refused {when}")
            except Refused as problem:
                assert reason in str(problem), problem

        old, new = reschedule(db, "741-333-333", tuesday.replace(hour=14), now)
        assert (old.hour, new.hour) == (9, 14) and len(free_slots(db, tuesday.date(), now)) == 15
        for call in (lambda: reschedule(db, "0000000000", tuesday.replace(hour=15), now), lambda: digits("12"),
                     lambda: parse("next tuesday", "9am")):
            try:
                call()
                raise AssertionError("should have refused")
            except Refused:
                pass
        assert say(new) == "Tuesday 24 November at 2:00 PM", say(new)
    print("appointment book ok")


if __name__ == "__main__":
    check() if "--check" in sys.argv else print(__doc__)
