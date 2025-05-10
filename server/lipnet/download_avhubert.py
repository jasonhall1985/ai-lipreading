import os
import sys
import logging
import requests
import subprocess
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AVHUBERT_REPO = "https://github.com/facebookresearch/av_hubert.git"
MODELS_DIR = Path("models/avhubert")
FAIRSEQ_DIR = Path("fairseq")

def download_file(url, dest_path, desc=None):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=desc,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def setup_environment():
    """Set up the necessary environment for AV-HuBERT"""
    logger.info("Setting up AV-HuBERT environment...")
    
    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clone AV-HuBERT repository if not exists
    if not Path("av_hubert").exists():
        logger.info("Cloning AV-HuBERT repository...")
        subprocess.run(["git", "clone", AVHUBERT_REPO], check=True)
        
        # Initialize and update submodules
        subprocess.run(["git", "submodule", "init"], cwd="av_hubert", check=True)
        subprocess.run(["git", "submodule", "update"], cwd="av_hubert", check=True)
    
    # Install dependencies
    logger.info("Installing dependencies...")
    subprocess.run(["pip", "install", "-r", "requirements_avhubert.txt"], check=True)
    
    # Install Fairseq
    if not FAIRSEQ_DIR.exists():
        logger.info("Installing Fairseq...")
        subprocess.run(["git", "clone", "https://github.com/pytorch/fairseq.git"], check=True)
        subprocess.run(["pip", "install", "--editable", "./"], cwd="fairseq", check=True)

def download_pretrained_models():
    """Download pre-trained AV-HuBERT models"""
    logger.info("Downloading pre-trained models...")
    
    # Model URLs (these would need to be replaced with actual URLs)
    model_urls = {
        "base_lrs3": "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_base.pt",
        "large_lrs3": "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_large.pt",
    }
    
    for model_name, url in model_urls.items():
        model_path = MODELS_DIR / f"{model_name}.pt"
        if not model_path.exists():
            logger.info(f"Downloading {model_name} model...")
            try:
                download_file(url, model_path, desc=f"Downloading {model_name}")
            except Exception as e:
                logger.error(f"Error downloading {model_name}: {str(e)}")
                continue
        else:
            logger.info(f"{model_name} model already exists")

def setup_configs():
    """Set up configuration files"""
    logger.info("Setting up configuration files...")
    
    config_dir = Path("configs/avhubert")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config files from AV-HuBERT repository
    subprocess.run([
        "cp",
        "av_hubert/conf/av_hubert/*.yaml",
        str(config_dir)
    ], check=True)

def main():
    """Main setup function"""
    try:
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("configs", exist_ok=True)
        
        # Setup steps
        setup_environment()
        download_pretrained_models()
        setup_configs()
        
        logger.info("AV-HuBERT setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during setup: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 