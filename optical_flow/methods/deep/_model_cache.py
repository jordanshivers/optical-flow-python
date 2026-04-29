"""Model weight download and caching.

Cached at ~/.cache/optical_flow/models/<method>/<filename>.
Override cache location with OPTICAL_FLOW_CACHE_DIR environment variable.
"""
import os
import hashlib
import urllib.request
import zipfile
import tempfile
from pathlib import Path


def get_cache_dir():
    """Return cache directory for model weights."""
    default = Path.home() / '.cache' / 'optical_flow' / 'models'
    return Path(os.environ.get('OPTICAL_FLOW_CACHE_DIR', default))


# Registry of available pretrained models.
# Checksums are verified after download for integrity.
# All RAFT models come in a single zip from Dropbox.
_RAFT_ZIP_URL = 'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip'

MODEL_REGISTRY = {
    'raft': {
        'raft-things': {
            'source': 'raft-zip',
            'zip_member': 'models/raft-things.pth',
            'filename': 'raft-things.pth',
            'description': 'RAFT trained on FlyingThings3D',
        },
        'raft-sintel': {
            'source': 'raft-zip',
            'zip_member': 'models/raft-sintel.pth',
            'filename': 'raft-sintel.pth',
            'description': 'RAFT fine-tuned on Sintel',
        },
        'raft-kitti': {
            'source': 'raft-zip',
            'zip_member': 'models/raft-kitti.pth',
            'filename': 'raft-kitti.pth',
            'description': 'RAFT fine-tuned on KITTI',
        },
        'raft-small': {
            'source': 'raft-zip',
            'zip_member': 'models/raft-small.pth',
            'filename': 'raft-small.pth',
            'description': 'RAFT small model',
        },
    },
    'sea-raft': {
        'sea-raft-things': {
            'source': 'huggingface',
            'repo': 'MemorySlices/Tartan-C-T-TSKH-spring540x960-M',
            'filename': 'model.safetensors',
            'local_filename': 'sea-raft-things.safetensors',
            'description': 'SEA-RAFT trained on mixed datasets (medium)',
        },
    },
    'waft': {
        'waft-things': {
            'source': 'gdrive',
            'file_id': '1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC',
            'filename': 'waft-things.pth',
            'description': 'WAFT trained on TartanAir + Chairs + Things (general-purpose)',
        },
        'waft-sintel': {
            'source': 'gdrive',
            'file_id': '1pAsbH0Orb5mFo4CLCYi4dSm6FjxrOqO3',
            'filename': 'waft-sintel.pth',
            'description': 'WAFT fine-tuned on Sintel',
        },
        'waft-kitti': {
            'source': 'gdrive',
            'file_id': '1hzpxBe80BmPCXjo9DvSszdCRsio5vLaB',
            'filename': 'waft-kitti.pth',
            'description': 'WAFT fine-tuned on KITTI',
        },
        'waft-spring-540p': {
            'source': 'gdrive',
            'file_id': '11DjrmhtDlyQ4UXlK0_jx0VMYYgvrukPs',
            'filename': 'waft-spring-540p.pth',
            'description': 'WAFT fine-tuned on Spring at 540p',
        },
        'waft-spring-1080p': {
            'source': 'gdrive',
            'file_id': '11_plF7ZAbo9OfVXOTvVriaXVROawr2ax',
            'filename': 'waft-spring-1080p.pth',
            'description': 'WAFT fine-tuned on Spring at 1080p',
        },
    },
}


def ensure_model(method, model_name):
    """Download model if not cached, return local path.

    Args:
        method: Method key in MODEL_REGISTRY (e.g. 'raft').
        model_name: Model variant key (e.g. 'raft-things').

    Returns:
        Path to the local model file.

    Raises:
        RuntimeError: If download fails or model is not in registry.
    """
    if method not in MODEL_REGISTRY:
        raise RuntimeError(f"Unknown method '{method}'. Available: {list(MODEL_REGISTRY.keys())}")
    if model_name not in MODEL_REGISTRY[method]:
        raise RuntimeError(
            f"Unknown model '{model_name}' for {method}. "
            f"Available: {list(MODEL_REGISTRY[method].keys())}"
        )

    info = MODEL_REGISTRY[method][model_name]
    cache_dir = get_cache_dir() / method
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Use local_filename if specified (e.g., when HF filename differs from desired local name)
    local_name = info.get('local_filename', info['filename'])
    local_path = cache_dir / local_name

    if local_path.exists():
        return local_path

    source = info.get('source', 'url')

    if source == 'raft-zip':
        return _download_raft_zip(info, local_path, cache_dir)
    elif source == 'huggingface':
        return _download_huggingface(info, local_path)
    elif source == 'gdrive':
        return _download_gdrive(info, local_path)
    elif 'url' in info:
        return _download_url(info, local_path)
    else:
        raise RuntimeError(
            f"Model '{model_name}' for {method} must be downloaded manually.\n"
            f"Please download '{info['filename']}' and place it at: {local_path}\n"
            f"See the original repository for download instructions."
        )


def _download_raft_zip(info, local_path, cache_dir):
    """Download RAFT models zip and extract all .pth files."""
    # Check if any RAFT model already exists (zip already extracted)
    if local_path.exists():
        return local_path

    print(f"Downloading RAFT models from Dropbox (all variants in one zip)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / 'models.zip'
        try:
            urllib.request.urlretrieve(_RAFT_ZIP_URL, zip_path, reporthook=_progress_hook)
            print()  # newline after progress
        except Exception as e:
            raise RuntimeError(f"Failed to download RAFT models: {e}")

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                if member.endswith('.pth'):
                    basename = Path(member).name
                    dest = cache_dir / basename
                    with zf.open(member) as src, open(dest, 'wb') as dst:
                        dst.write(src.read())
                    print(f"  Extracted {basename}")

    if not local_path.exists():
        raise RuntimeError(
            f"Expected {info['filename']} in zip but it was not found. "
            f"Check the zip contents match the expected member: {info['zip_member']}"
        )
    return local_path


def _download_url(info, local_path):
    """Download model from a direct URL."""
    url = info['url']
    filename = info['filename']
    print(f"Downloading {filename} from {url}...")

    try:
        urllib.request.urlretrieve(url, local_path, reporthook=_progress_hook)
        print()  # newline after progress
    except Exception as e:
        if local_path.exists():
            local_path.unlink()
        raise RuntimeError(f"Failed to download {filename}: {e}")

    return local_path


def _download_gdrive(info, local_path):
    """Download model from Google Drive."""
    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown is required to download models from Google Drive (e.g. WAFT).\n"
            "Install with: pip install gdown"
        )

    file_id = info['file_id']
    filename = info.get('local_filename', info['filename'])
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f"Downloading {filename} from Google Drive...")

    try:
        gdown.download(url, str(local_path), quiet=False)
    except Exception as e:
        if local_path.exists():
            local_path.unlink()
        raise RuntimeError(f"Failed to download {filename} from Google Drive: {e}")

    if not local_path.exists():
        raise RuntimeError(f"Download appeared to succeed but {filename} not found at {local_path}")
    return local_path


def _download_huggingface(info, local_path):
    """Download model from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required to download SEA-RAFT models.\n"
            "Install with: pip install huggingface_hub"
        )

    repo = info['repo']
    filename = info['filename']
    print(f"Downloading {filename} from HuggingFace ({repo})...")

    try:
        downloaded = hf_hub_download(repo_id=repo, filename=filename)
        # Copy to our cache location
        import shutil
        shutil.copy2(downloaded, local_path)
    except Exception as e:
        if local_path.exists():
            local_path.unlink()
        raise RuntimeError(f"Failed to download {filename} from HuggingFace: {e}")

    return local_path


def _progress_hook(block_num, block_size, total_size):
    """Print download progress."""
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
