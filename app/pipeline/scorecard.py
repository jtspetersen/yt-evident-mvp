# app/pipeline/scorecard.py

def tally(verdicts):
    """Count verdicts by rating and collect red flags and source tiers."""
    counts = {"VERIFIED":0, "LIKELY TRUE":0, "INSUFFICIENT EVIDENCE":0, "CONFLICTING EVIDENCE":0, "LIKELY FALSE":0, "FALSE":0}
    red_flags = {}
    tiers = {}

    for v in verdicts:
        counts[v.rating] += 1
        for rf in v.red_flags:
            red_flags[rf] = red_flags.get(rf, 0) + 1
        for t in v.source_tiers_used:
            tiers[t] = tiers.get(t, 0) + 1

    return counts, red_flags, tiers
