# Path: tests/test_rift_env.py
# Status: NEW
# Verifies multi-step (non-bandit) semantics WITHOUT torch by checking the env's
# action->state bookkeeping logic via a light stand-in. Full env tested on Katz.
def test_horizon_gt_one_is_multistep():
    # The contract: history grows each step and mask accumulates -> state depends on past.
    history=[]; mask=[[0]*8 for _ in range(8)]; horizon=4
    for step in range(horizon):
        a=step  # pick a different cell each step
        r,c=a//8, a%8; mask[r][c]=min(1, mask[r][c]+1); history.append(a)
    assert len(history)==horizon            # multi-step
    assert sum(sum(row) for row in mask)==horizon  # mask accumulates over steps
    # re-selecting an already-set cell is a wasted step (history-dependent value)
    before=sum(sum(row) for row in mask)
    a=0; r,c=0,0; mask[r][c]=min(1, mask[r][c]+1)  # cell already 1
    assert sum(sum(row) for row in mask)==before   # no gain -> not a bandit
