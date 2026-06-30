# Path: tests/test_identity_gap_adapter.py
# Status: NEW
import warnings, pytest
from iganer.rift.adapters.identity_gap_contract import (
    resolve_mode, IdentityGapMode, IdentityGapResult, MechanismValidityError)
def test_true_mode(): assert resolve_mode(True, False)==IdentityGapMode.TRUE
def test_error_mode(): assert resolve_mode(False, True)==IdentityGapMode.ERROR
def test_proxy_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert resolve_mode(False, False)==IdentityGapMode.PROXY
        assert any("PROXY" in str(x.message) for x in w)
def test_mechanism_guard_raises():
    r=IdentityGapResult(0.5, IdentityGapMode.PROXY, False)
    with pytest.raises(MechanismValidityError): r.assert_mechanism_valid()
