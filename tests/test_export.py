import json

import numpy as np
import pandas as pd

from xpfpl import export


def test_table_is_column_wise_and_json_safe():
    df = pd.DataFrame({"a": [1, 2], "x": [0.123456, np.nan], "t": pd.to_datetime(["2026-08-16", None], utc=True),
                       "b": [True, False]})
    out = export.table(df, digits=3)
    assert out == {"a": [1, 2], "x": [0.123, None], "t": ["2026-08-16T00:00:00+00:00", None], "b": [True, False]}
    json.dumps(out, allow_nan=False)


def test_write_scrubs_nan_from_nested_reports(tmp_path):
    path = export._write({"ranking": [{"spearman": float("nan")}, {"spearman": 0.5}], "inf": float("inf")},
                         tmp_path / "r.json")
    assert json.loads(path.read_text()) == {"ranking": [{"spearman": None}, {"spearman": 0.5}], "inf": None}


def test_saved_forecasts_prefer_the_chosen_model(tmp_path, monkeypatch):
    from xpfpl.data import archive
    monkeypatch.setattr(archive, "PREDICTIONS", tmp_path)
    for model, xp in (("components", 1.0), ("mlp", 2.0)):
        archive.write(pd.DataFrame({"element": [7], "xp_6": [xp], "xp_7": [0.5]}), tmp_path / "2026-27" / f"gw06_{model}.parquet")
    archive.write(pd.DataFrame({"element": [7], "xp_7": [3.0]}), tmp_path / "2026-27" / "gw07_components.parquet")
    from xpfpl import review
    got = review.saved_forecasts("2026-27", "mlp")
    assert got[6].loc[7] == 2.0 and got[6].name == "mlp"
    assert got[7].loc[7] == 3.0            # only another model's forecast was saved: use it


def test_seal_opens_only_with_the_secret(monkeypatch):
    import base64

    import pytest
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    monkeypatch.setattr(export, "SEAL_ITERATIONS", 1000)
    sealed = export.seal({"captain": 12, "xp": float("nan")}, " three blind mice ")

    def unseal(word: str):
        raw = lambda k: base64.b64decode(sealed[k])   # noqa: E731
        key = PBKDF2HMAC(hashes.SHA256(), 32, raw("salt"), sealed["iterations"]).derive(word.strip().encode())
        return json.loads(AESGCM(key).decrypt(raw("iv"), raw("data"), None))

    assert sealed["v"] == 1 and "captain" not in json.dumps(sealed)
    assert unseal("three blind mice") == {"captain": 12, "xp": None}
    with pytest.raises(InvalidTag):
        unseal("three blind rats")
