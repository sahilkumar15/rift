# Path: src/audit/leaderboard.py
# Status: NEW
"""Format aggregate audit rows into a sorted leaderboard (by faithfulness_delta)."""
def build_leaderboard(agg_rows, sort_key="faithfulness_delta"):
    rows=[r for r in agg_rows if r]
    rows.sort(key=lambda r: r.get(sort_key, 0.0), reverse=True)
    cols=["explainer","identity_gap_mode","faithfulness_delta","faithfulness_logit",
          "necessity_delta","sufficiency_delta","plausibility_iou","mask_area",
          "identity_preservation","perceptual_distance","n"]
    table=[[r.get(c) for c in cols] for r in rows]
    return cols, table
