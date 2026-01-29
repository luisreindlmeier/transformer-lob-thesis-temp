#!/usr/bin/env python3
"""
Event-Type-Verteilung pro Ticker (absolut), 4 Ticker in 4 Spalten.
Im Notebook nach der Table-4.2-Zelle ausführen (raw_stats muss existieren).
Oder: In einer Notebook-Zelle einfügen und ausführen.
"""
import pandas as pd

EVENT_TYPE_NAMES = {
    1: 'Limit Order Submission',
    2: 'Partial Cancellation',
    3: 'Full Cancellation',
    4: 'Execution (Visible)',
    5: 'Execution (Hidden)',
    6: 'Cross Trade',
    7: 'Trading Halt',
}

TICKERS = ["AAPL", "CSCO", "GOOG", "INTC"]

def event_counts_per_ticker(raw_stats):
    if not raw_stats:
        print("raw_stats ist leer.")
        return
    all_events = set()
    for d in raw_stats.values():
        all_events.update(d.get("event_type_counts", {}).keys())
    rows = []
    for et in sorted(all_events):
        row = {"Event Type": et, "Event Name": EVENT_TYPE_NAMES.get(et, f"Type {et}")}
        for t in TICKERS:
            row[t] = raw_stats.get(t, {}).get("event_type_counts", {}).get(et, 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.set_index(["Event Type", "Event Name"])
    return df

# --- Im Notebook: raw_stats existiert bereits. Einfach ausführen:
# df_events_pivot = event_counts_per_ticker(raw_stats)
# print(df_events_pivot)
# display(df_events_pivot)  # optional

if __name__ == "__main__":
    # Standalone: nur ausführbar, wenn raw_stats geladen wird (z.B. aus Notebook-Kernel)
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and "raw_stats" in ip.user_ns:
            df_events_pivot = event_counts_per_ticker(ip.user_ns["raw_stats"])
            print(df_events_pivot)
        else:
            print("raw_stats nicht gefunden. Bitte Skript in einer Notebook-Zelle nach der Table-4.2-Zelle ausführen.")
    except Exception:
        print("raw_stats nicht gefunden. Bitte folgenden Code in einer Notebook-Zelle nach der Table-4.2-Zelle ausführen:")
        print("""
df_events_pivot = event_counts_per_ticker(raw_stats)
print(df_events_pivot)
# display(df_events_pivot)
""")
