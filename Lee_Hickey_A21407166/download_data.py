import os
from pathlib import Path
import kagglehub

def download_creditcard_dataset(output_dir="data"):
    # Ensure ./data folder exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("Downloading dataset using KaggleHub...")
    hub_path = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud"))
    print("KaggleHub stored files at:", hub_path)

    # Move KaggleHub files into ./data using Pathlib
    for file_path in hub_path.rglob("*"):
        if file_path.is_file():
            destination = output_path / file_path.name
            destination.write_bytes(file_path.read_bytes())  

    print("All files copied to working directory:", output_path)


if __name__ == "__main__":
    download_creditcard_dataset()
