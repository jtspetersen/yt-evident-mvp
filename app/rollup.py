# app/rollup.py
import argparse
from app.store.creator_rollup import load_creator_events, rollup_by_channel, summarize_channel

def main():
    parser = argparse.ArgumentParser(description="Creator rollups from store/creator_profiles.jsonl")
    parser.add_argument("--channel", type=str, default=None, help="Only show this channel (exact match).")
    parser.add_argument("--top", type=int, default=10, help="Show top N channels by run count.")
    args = parser.parse_args()

    events = load_creator_events()
    by = rollup_by_channel(events)

    if args.channel:
        ch = args.channel.strip()
        ev = by.get(ch, [])
        if not ev:
            print(f"No events found for channel: {ch}")
            return
        s = summarize_channel(ev)
        print_channel(ch, s)
        return

    # sort channels by run count desc
    items = sorted(by.items(), key=lambda kv: len(kv[1]), reverse=True)[:int(args.top)]
    for ch, ev in items:
        s = summarize_channel(ev)
        print_channel(ch, s)
        print()

def print_channel(ch, s):
    print(f"Channel: {ch}")
    print(f"  Runs: {s['runs']}")
    if s["verdict_totals"]:
        print(f"  Verdict totals: {s['verdict_totals']}")
    if s["top_red_flags"]:
        print("  Top red flags:")
        for k, v in s["top_red_flags"]:
            print(f"    - {k}: {v}")
    if s["top_topics"]:
        print("  Top topics:")
        for k, v in s["top_topics"]:
            print(f"    - {k}: {v}")

    print("  Recent runs:")
    for r in s["recent"]:
        ts = r.get("timestamp") or "?"
        rid = r.get("run_id") or "?"
        inp = r.get("input_file") or "?"
        print(f"    - {ts} | {rid} | {inp}")

if __name__ == "__main__":
    main()
