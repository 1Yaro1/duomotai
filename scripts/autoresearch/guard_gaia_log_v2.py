"""Additional sidecar guard. Does not replace/relax guard_gaia_protocol.py."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source", {}).get("source_sha256") != "fe457cf7c65a1b7928a06959f9a245d6bd384e38dbc2804e47df63791d0c2888":
        raise ValueError("Not the verified original GAIA business archive (or a synthetic fixture)")
    semantic = manifest.get("semantics", {})
    if semantic.get("type") != "bert" or semantic.get("frozen") is not True or not semantic.get("files_sha256"):
        raise ValueError("Missing frozen BERT provenance; synthetic embeddings are not valid experiment inputs")
    path = Path(__file__).resolve().parents[2] / "ts_benchmark/baselines/MindTS/cmc_log_data.py"
    spec = importlib.util.spec_from_file_location("log_v2_guard_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    preparer = Path(__file__).resolve().with_name("prepare_gaia_log_features_v2.py")
    if manifest.get("preparer_sha256") != module.digest(preparer) or manifest.get("drain3_version") != "0.9.11":
        raise ValueError("Preprocessing implementation/version differs from cache provenance")
    for service in module.SERVICES:
        store = module.LogV2Store(args.cache_dir, service)
        print(json.dumps({"service": service, "rows": len(store.counts),
                          "templates_including_unknown": store.count_dim,
                          "semantic_dim": store.semantic_dim}))
    print(json.dumps({"log_v2_guard": "ok", "scope": "train_validation_only"}))


if __name__ == "__main__":
    main()
