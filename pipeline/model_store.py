"""
pipeline/model_store.py
Save and load the trained XGBoost model via Hugging Face Hub (free model storage).

The model is stored as a pickled file in a HF dataset/model repo.
On each retrain, the old model is overwritten and the new commit SHA becomes the version tag.
"""

import logging
import pickle
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Local cache path so we don't re-download every prediction
_LOCAL_CACHE = Path(__file__).resolve().parent.parent / "models" / "xgb_model.pkl"


# ─── Push ─────────────────────────────────────────────────────────────────────

def push_model(model, feature_cols: list[str], metrics: dict) -> str:
    """
    Serialise model + metadata, upload to Hugging Face Hub.

    Returns the commit SHA (used as model version tag).
    """
    from huggingface_hub import HfApi, CommitOperationAdd
    from config import HF_TOKEN, HF_USERNAME, HF_REPO_NAME, HF_MODEL_FILENAME

    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"

    # Ensure repo exists (creates it if not)
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, token=HF_TOKEN,
                        repo_type="model", exist_ok=True, private=False)
    except Exception as exc:
        logger.warning("Repo creation skipped (may already exist): %s", exc)

    # Serialise model bundle
    bundle = {
        "model":        model,
        "feature_cols": feature_cols,
        "metrics":      metrics,
    }
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump(bundle, tmp)
        tmp_path = Path(tmp.name)

    # Upload
    try:
        commit_info = api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=HF_MODEL_FILENAME,
            repo_id=repo_id,
            token=HF_TOKEN,
            commit_message=f"Retrain — MAE={metrics.get('mae', '?'):.2f}",
        )
        sha = commit_info.oid or commit_info.commit_url.split("/")[-1]
        logger.info("Model pushed to %s  (commit %s)", repo_id, sha[:8])
        return sha
    finally:
        tmp_path.unlink(missing_ok=True)


# ─── Pull ─────────────────────────────────────────────────────────────────────

def load_model(force_download: bool = False):
    """
    Load the model bundle from local cache, or download from HF Hub if missing.

    Returns (model, feature_cols, metrics) tuple.
    Raises FileNotFoundError if neither cache nor HF Hub has the model.
    """
    if not force_download and _LOCAL_CACHE.exists():
        logger.info("Loading model from local cache: %s", _LOCAL_CACHE)
        return _read_bundle(_LOCAL_CACHE)

    return _download_from_hub()


def _download_from_hub():
    """Download model file from HF Hub, cache it, and return the bundle."""
    from huggingface_hub import hf_hub_download
    from config import HF_TOKEN, HF_USERNAME, HF_REPO_NAME, HF_MODEL_FILENAME

    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"
    logger.info("Downloading model from %s …", repo_id)

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=HF_MODEL_FILENAME,
            token=HF_TOKEN,
        )
        _LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(local_path, _LOCAL_CACHE)
        logger.info("Model cached → %s", _LOCAL_CACHE)
        return _read_bundle(_LOCAL_CACHE)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not load model from HF Hub ({repo_id}): {exc}\n"
            "Run pipeline/train.py first to train and push the initial model."
        ) from exc


def _read_bundle(path: Path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    model       = bundle["model"]
    feature_cols = bundle["feature_cols"]
    metrics     = bundle.get("metrics", {})
    return model, feature_cols, metrics


def save_local(model, feature_cols: list[str], metrics: dict) -> Path:
    """Save model bundle to local cache only (no HF upload). Used during training."""
    _LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "feature_cols": feature_cols, "metrics": metrics}
    with open(_LOCAL_CACHE, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Model saved locally → %s", _LOCAL_CACHE)
    return _LOCAL_CACHE
