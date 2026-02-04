# app/pipeline/scorecard.py
def severity_weight(sev: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(sev, 2)

def score(verdicts):
    s = 100
    counts = {"VERIFIED":0, "LIKELY TRUE":0, "UNCERTAIN":0, "LIKELY FALSE":0, "FALSE":0}
    red_flags = {}
    tiers = {}

    for v in verdicts:
        counts[v.rating] += 1
        w = severity_weight(v.severity)
        if v.rating == "FALSE":
            s -= 6 * w
        elif v.rating == "LIKELY FALSE":
            s -= 4 * w
        elif v.rating == "UNCERTAIN":
            s -= 2 * w

        for rf in v.red_flags:
            red_flags[rf] = red_flags.get(rf, 0) + 1
        for t in v.source_tiers_used:
            tiers[t] = tiers.get(t, 0) + 1

    s = max(0, min(100, s))
    return s, counts, red_flags, tiers
