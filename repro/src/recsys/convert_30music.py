"""Convert the 30Music ``events.idomaar`` play-event log to the flat CSV
expected by Klenitskiy et al. 2024 (ref [25], github.com/Antondfger/Does-It-Look-Sequential).

Their pipeline reads ``30M.csv`` with columns ``user,item,timestamp`` (epoch
seconds; ``time_coding: 's'`` in their 30Music.yaml). The raw dataset ships in
idomaar format, one play event per line::

    event.play  <id>  <unix_ts>  {"playtime":176}  {"subjects":[{"type":"user","id":U}], "objects":[{"type":"track","id":T}]}

We keep events in file order (session order, chronological within a session);
every downstream step of [25] re-sorts by (user, timestamp), so only
same-second tie order depends on this, and file order is the natural
chronological construction.

CLI::

    python -m src.recsys.convert_30music --events /path/relations/events.idomaar \
        --out /path/30M.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys

LINE_RE = re.compile(
    rb'^event\.play\t\d+\t(-?\d+)\t[^\t]*\t.*?"user","id":(\d+).*?"track","id":(\d+)'
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(events_path: str, out_path: str) -> dict:
    total = parsed = 0
    with open(events_path, "rb") as fin, open(out_path, "wb") as fout:
        fout.write(b"user,item,timestamp\n")
        buf = []
        for line in fin:
            total += 1
            m = LINE_RE.match(line)
            if m is None:
                continue
            ts, user, track = m.group(1), m.group(2), m.group(3)
            buf.append(user + b"," + track + b"," + ts + b"\n")
            parsed += 1
            if len(buf) >= 100_000:
                fout.write(b"".join(buf))
                buf.clear()
        fout.write(b"".join(buf))
    return {"total_lines": total, "parsed": parsed, "skipped": total - parsed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stats = convert(args.events, args.out)
    stats["events_idomaar_sha256"] = sha256_file(args.events)
    stats["out_sha256"] = sha256_file(args.out)
    with open(args.out + ".convert_meta.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(stats)
    if stats["parsed"] == 0 or stats["skipped"] > 0:
        sys.exit(f"FATAL: parsed={stats['parsed']} skipped={stats['skipped']} — "
                 "regex assumptions violated, refusing to ship")


if __name__ == "__main__":
    main()
