import os
import urllib.request
import zipfile
import tarfile
import argparse
from tqdm import tqdm

GRID_URLS = {
    's1': 'http://spandh.dcs.shef.ac.uk/gridcorpus/s1/video/s1.mpg_vcd.zip',
    's2': 'http://spandh.dcs.shef.ac.uk/gridcorpus/s2/video/s2.mpg_vcd.zip',
    's3': 'http://spandh.dcs.shef.ac.uk/gridcorpus/s3/video/s3.mpg_vcd.zip',
    's4': 'http://spandh.dcs.shef.ac.uk/gridcorpus/s4/video/s4.mpg_vcd.zip',
}

ALIGN_URLS = {
    's1': 'http://spandh.dcs.shef.ac.uk/gridcorpus/s1/align/s1.tar',
    's2': 'http://spandh.dcs.shef.ac.uk/gridcorpus/s2/align/s2.tar',
    's3': 'http://spandh.dcs.shef.ac.uk/gridcorpus/s3/align/s3.tar',
    's4': 'http://spandh.dcs.shef.ac.uk/gridcorpus/s4/align/s4.tar',
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download a file from URL to output_path with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_grid_corpus(output_dir, speakers=None):
    """
    Download GRID corpus data
    
    Args:
        output_dir: Directory to save the dataset
        speakers: List of speaker IDs to download (e.g. ['s1', 's2']). If None, downloads all.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # If no speakers specified, download all
    if speakers is None:
        speakers = list(GRID_URLS.keys())
    
    # Download video data
    print("Downloading video data...")
    for speaker in speakers:
        if speaker not in GRID_URLS:
            print(f"Warning: Speaker {speaker} not found in URLs")
            continue
            
        video_url = GRID_URLS[speaker]
        align_url = ALIGN_URLS[speaker]
        
        # Create speaker directory
        speaker_dir = os.path.join(output_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        
        # Download video zip
        video_path = os.path.join(speaker_dir, f"{speaker}_video.zip")
        if not os.path.exists(video_path):
            print(f"\nDownloading videos for {speaker}...")
            download_url(video_url, video_path)
            
            # Extract video zip
            print(f"Extracting videos for {speaker}...")
            with zipfile.ZipFile(video_path, 'r') as zip_ref:
                zip_ref.extractall(speaker_dir)
        
        # Download alignment file
        align_path = os.path.join(speaker_dir, f"{speaker}_align.tar")
        if not os.path.exists(align_path):
            print(f"\nDownloading alignments for {speaker}...")
            download_url(align_url, align_path)
            
            # Extract alignment tar
            print(f"Extracting alignments for {speaker}...")
            with tarfile.open(align_path, 'r') as tar_ref:
                tar_ref.extractall(speaker_dir)
    
    print("\nDownload complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GRID corpus dataset")
    parser.add_argument("--output_dir", default="grid_data",
                      help="Directory to save the dataset")
    parser.add_argument("--speakers", nargs="+", default=None,
                      help="List of speakers to download (e.g. s1 s2)")
    
    args = parser.parse_args()
    download_grid_corpus(args.output_dir, args.speakers) 