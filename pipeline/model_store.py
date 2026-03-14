"""
pipeline/model_store.py
Save and load the trained XGBoost model via Hugging Face Hub (free model storage).

The model is stored using XGBoost's native binary format (safe, no arbitrary code execution).
Metadata (feature columns, metrics) stored as JSON.
On each retrain, the old model is overwritten and the new commit SHA becomes the version tag.
"""

import logging
import json
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Local cache paths so we don't re-download every prediction
_LOCAL_CACHE_DIR = Path(__file__).resolve().parent.parent / "models"
_LOCAL_MODEL = _LOCAL_CACHE_DIR / "xgb_model.bin"   # XGBoost native binary format
_LOCAL_META  = _LOCAL_CACHE_DIR / "model_metadata.json"  # Feature cols + metrics


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_serialisable(v):
    """Return a JSON-safe representation of v.

    Handles the fact that the metrics dict from train.py contains non-float
    values such as best_params (dict), feature_cols (list), and
    top_features (list of tuples) that would crash float().
    """
    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _make_serialisable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_make_serialisable(item) for item in v]
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


# ─── Push ─────────────────────────────────────────────────────────────────────

def push_model(model, feature_cols: list[str], metrics: dict) -> str:
    """
    Serialise model + metadata, upload to Hugging Face Hub.
    Uses XGBoost native format (safe, no code execution) + JSON metadata.

    Returns the commit SHA (used as model version tag).
    """
    from huggingface_hub import HfApi, CommitOperationAdd
    from config import HF_TOKEN, HF_USERNAME, HF_REPO_NAME

    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"

    # Ensure repo exists (creates it if not)
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, token=HF_TOKEN,
                        repo_type="model", exist_ok=True, private=True)
    except Exception as exc:
        logger.warning("Repo creation skipped (may already exist): %s", exc)

    # Save model using XGBoost native format (binary, safe)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_path = Path(tmpdir) / "model.bin"
        tmp_meta_path  = Path(tmpdir) / "metadata.json"

        # Model: XGBoost native binary format (works for both XGBRegressor and Booster)
        model.save_model(str(tmp_model_path))

        # Metadata: JSON format — use _make_serialisable to handle nested dicts/lists
        metadata = {
            "feature_cols": feature_cols,
            "metrics": {k: _make_serialisable(v) for k, v in metrics.items()},
        }
        with open(tmp_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Upload both files
        try:
            # Upload model
            api.upload_file(
                path_or_fileobj=tmp_model_path,
                path_in_repo="model.bin",
                repo_id=repo_id,
                token=HF_TOKEN,
                commit_message=f"Retrain — MAE={metrics.get('mae', 'N/A')}",
            )
            # Upload metadata (capture commit info for the version SHA)
            commit_info = api.upload_file(
                path_or_fileobj=tmp_meta_path,
                path_in_repo="metadata.json",
                repo_id=repo_id,
                token=HF_TOKEN,
                commit_message=f"Retrain — MAE={metrics.get('mae', 'N/A')}",
            )
            # Extract commit SHA safely
            sha = commit_info.commit_url.split("/")[-1] if commit_info.commit_url else "unknown"
            logger.info("Model pushed to %s (commit %s)", repo_id, sha[:8])
            return sha
        except Exception as exc:
            logger.error("Failed to push model to HF Hub: %s", exc)
            raise


# ─── Pull ─────────────────────────────────────────────────────────────────────

def load_model(force_download: bool = False):
    """
    Load the model from local cache, or download from HF Hub if missing.

    Returns (model, feature_cols, metrics) tuple.
    Uses XGBoost native format loaded into XGBRegressor (sklearn API) so that
    model.predict(numpy_array) works directly in daily_job.py without DMatrix.

    Raises FileNotFoundError if neither cache nor HF Hub has the model.
    """
    if not force_download and _LOCAL_MODEL.exists() and _LOCAL_META.exists():
        logger.info("Loading model from local cache")
        return _read_bundle_local()

    return _download_from_hub()


def _read_bundle_local():
    """Load model + metadata from local cache, returning an XGBRegressor."""
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(str(_LOCAL_MODEL))

    with open(_LOCAL_META, "r") as f:
        meta = json.load(f)

    return model, meta["feature_cols"], meta.get("metrics", {})


def _download_from_hub():
    """Download model files from HF Hub, cache them, and return the bundle."""
    import xgboost as xgb
    from huggingface_hub import hf_hub_download
    from config import HF_TOKEN, HF_USERNAME, HF_REPO_NAME

    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"
    logger.info("Downloading model from %s …", repo_id)

    try:
        # Download model binary
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.bin",
            token=HF_TOKEN,
        )
        # Download metadata
        meta_path = hf_hub_download(
            repo_id=repo_id,
            filename="metadata.json",
            token=HF_TOKEN,
        )

        # Cache both files locally
        _LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(model_path, _LOCAL_MODEL)
        shutil.copy(meta_path,  _LOCAL_META)
        logger.info("Model cached at %s", _LOCAL_CACHE_DIR)

        # Load from cache using XGBRegressor (sklearn API)
        model = xgb.XGBRegressor()
        model.load_model(str(_LOCAL_MODEL))

        with open(_LOCAL_META, "r") as f:
            meta = json.load(f)

        return model, meta["feature_cols"], meta.get("metrics", {})

    except Exception as exc:
        raise FileNotFoundError(
            f"Could not load model from HF Hub ({repo_id}): {exc}\n"
            "Run pipeline/train.py first to train and push the initial model."
        ) from exc


def save_local(model, feature_cols: list[str], metrics: dict) -> Path:
    """Save model + metadata to local cache only (no HF upload). Used during training."""
    _LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Save model using XGBoost native binary format
    model.save_model(str(_LOCAL_MODEL))

    # Save metadata as JSON — use _make_serialisable for nested types
    metadata = {
        "feature_cols": feature_cols,
        "metrics": {k: _make_serialisable(v) for k, v in metrics.items()},
    }
    with open(_LOCAL_META, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model saved locally → %s", _LOCAL_MODEL)
    return _LOCAL_MODEL
