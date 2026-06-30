# Path: tests/test_interventions.py
# Status: NEW
import numpy as np
# numpy mirror of mask complementarity (torch tested on Katz)
def to_binary(mask, tf):
    flat=mask.reshape(mask.shape[0],-1); k=max(1,int(tf*flat.shape[1]))
    out=np.zeros_like(flat)
    for i in range(flat.shape[0]):
        out[i, np.argsort(flat[i])[::-1][:k]]=1.0
    return out.reshape(mask.shape)
def test_nec_suf_complement():
    np.random.seed(0); img=np.random.rand(2,1,8,8); m=np.random.rand(2,1,8,8); tf=0.25
    b=to_binary(m,tf); nec=img*(1-b); suf=img*b
    assert np.allclose(nec+suf, img)
def test_mask_area():
    np.random.seed(1); m=np.random.rand(2,1,8,8)
    assert abs(to_binary(m,0.25).mean()-0.25)<0.03
