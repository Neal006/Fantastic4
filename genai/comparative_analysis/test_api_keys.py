"""
Test all API keys to verify they are working before running the ablation study.

Usage:
    cd genai
    python -m comparative_analysis.test_api_keys
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comparative_analysis.config import MODELS
from comparative_analysis.model_clients import create_client


def test_all_keys():
    print("=" * 65)
    print("  LUMIN.AI  –  API Key Verification")
    print("=" * 65)
    print()

    results = {}
    all_ok = True

    for model_key, model_cfg in MODELS.items():
        display = model_cfg["display_name"]
        provider = model_cfg["provider"]
        has_key = bool(model_cfg["api_key"])

        print(f"  [{provider.upper()}] {display}")
        print(f"    Model ID : {model_cfg['model_id']}")
        print(f"    API Key  : {'***' + model_cfg['api_key'][-6:] if has_key else '✗ MISSING'}")

        if not has_key:
            print(f"    Status   : ✗ SKIPPED – no API key\n")
            results[model_key] = {"ok": False, "message": "No API key configured"}
            all_ok = False
            continue

        try:
            client = create_client(model_cfg)
            health = client.health_check()

            # Treat 429 rate-limit as "key valid but temporarily limited"
            is_rate_limit = "429" in health.get("message", "")
            if health["ok"]:
                results[model_key] = health
                print(f"    Status   : ✓ CONNECTED")
                print(f"    Response : {health['message'][:60]}")
            elif is_rate_limit:
                results[model_key] = {"ok": True, "message": "Rate-limited (429) – key is valid"}
                print(f"    Status   : ⚠ RATE-LIMITED (429) – key valid, temporarily throttled")
            else:
                results[model_key] = health
                print(f"    Status   : ✗ FAILED")
                print(f"    Error    : {health['message'][:100]}")
                all_ok = False

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str
            if is_rate_limit:
                results[model_key] = {"ok": True, "message": "Rate-limited (429) – key is valid"}
                print(f"    Status   : ⚠ RATE-LIMITED (429) – key valid, temporarily throttled")
            else:
                results[model_key] = {"ok": False, "message": err_str}
                print(f"    Status   : ✗ ERROR")
                print(f"    Error    : {err_str[:100]}")
                all_ok = False

        print()
        time.sleep(4)  # avoid rate-limiting between checks

    # Summary
    print("=" * 65)
    ok_count = sum(1 for r in results.values() if r["ok"])
    total = len(results)
    print(f"  Results: {ok_count}/{total} models connected successfully")

    if all_ok:
        print("  ✓ All API keys are working! Ready to run ablation study.")
    else:
        failed = [k for k, r in results.items() if not r["ok"]]
        print(f"  ✗ Failed models: {', '.join(failed)}")
        print("  Fix the above issues or remove failing models from config.py")

    print("=" * 65)
    return all_ok


if __name__ == "__main__":
    success = test_all_keys()
    sys.exit(0 if success else 1)
