# app/tools/review.py
def _prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except EOFError:
        return "q"

def review_claims_interactive(claims):
    """
    Interactive selection/edit loop.
    Returns filtered + edited list of claims (same objects, mutated safely).
    """
    kept = []
    total = len(claims)

    print("\n=== REVIEW MODE ===")
    print("Commands: [y]=keep  [n]=drop  [e]=edit text  [t]=edit type  [s]=skip/no change  [q]=quit review\n")

    for i, c in enumerate(claims, start=1):
        print("-" * 72)
        print(f"[{i}/{total}] claim_id={getattr(c, 'claim_id', '?')}")
        print(f"type:  {getattr(c, 'claim_type', '')}")
        print(f"text:  {getattr(c, 'claim_text', '')}")
        q = getattr(c, "quote_from_transcript", "") or ""
        if q:
            print(f"quote: {q[:240]}{'...' if len(q) > 240 else ''}")

        while True:
            cmd = _prompt("Action (y/n/e/t/s/q): ").lower()

            if cmd in ("y", "keep"):
                kept.append(c)
                break

            if cmd in ("n", "drop"):
                break

            if cmd in ("s", "skip"):
                # default is keep (since extracted claims were chosen intentionally)
                kept.append(c)
                break

            if cmd in ("q", "quit"):
                print("\nReview quit early.")
                return kept

            if cmd in ("e", "edit"):
                new_text = _prompt("New claim text (leave blank to keep): ")
                if new_text:
                    c.claim_text = new_text
                new_quote = _prompt("New quote (leave blank to keep): ")
                if new_quote:
                    c.quote_from_transcript = new_quote
                # after edit, ask keep/drop
                confirm = _prompt("Keep this claim? (y/n): ").lower()
                if confirm.startswith("y"):
                    kept.append(c)
                break

            if cmd in ("t", "type"):
                new_type = _prompt("New claim type (e.g., statistical, medical, causal, historical, other): ")
                if new_type:
                    c.claim_type = new_type
                confirm = _prompt("Keep this claim? (y/n): ").lower()
                if confirm.startswith("y"):
                    kept.append(c)
                break

            print("Unknown command. Use y/n/e/t/s/q.")

    return kept
