# Path: tests/test_faithfulness_metrics.py
# Status: NEW
import math
from iganer.rift.faithfulness.faithfulness_score import (
    necessity, sufficiency, harmonic, compute_rift_score)
def test_perfect_faithful():
    assert abs(necessity(1.0,0.0)-1.0)<1e-6
    assert abs(sufficiency(1.0,1.0)-1.0)<1e-6
    assert harmonic(1.0,1.0)>0.99
def test_mare_failure_mode_scores_low():
    nec=necessity(1.0,0.9); suf=sufficiency(1.0,0.95)
    assert harmonic(nec,suf) < 0.35
def test_proxy_mode_strips_delta_credit():
    t=compute_rift_score(e0_delta=1,e_nec_delta=0,e_suf_delta=1,
        e0_logit=2,e_nec_logit=0,e_suf_logit=2,mask_area=0.1,identity_gap_mode="true")
    p=compute_rift_score(e0_delta=1,e_nec_delta=0,e_suf_delta=1,
        e0_logit=2,e_nec_logit=0,e_suf_logit=2,mask_area=0.1,identity_gap_mode="proxy")
    assert p.rift_score < t.rift_score
def test_all_finite():
    c=compute_rift_score(e0_delta=0.5,e_nec_delta=0.2,e_suf_delta=0.4,
        e0_logit=1,e_nec_logit=0.3,e_suf_logit=0.8,mask_area=0.2,identity_gap_mode="true")
    for v in c.to_dict().values():
        if isinstance(v,float): assert math.isfinite(v)
