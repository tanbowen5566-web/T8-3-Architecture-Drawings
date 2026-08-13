from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

# Define a function to load one dataset 
def load_dataset(dataset_name):
    dataset_path = DATA_DIR / dataset_name
    metadata_path = dataset_path / "metadata.csv"

    # Check whether the dataset folder exists
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}"
        )

    # Check whether metadata.csv exists
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )
    
    # Read metadata.csv
    metadata = pd.read_csv(metadata_path)

    # Check whether the Node ID column exists
    if "Node ID" not in metadata.columns:
         raise ValueError(
            "Node ID column is missing from metadata.csv"
        )
    
    # Store matched "image + metadata"
    records = []

    # Read the metadata one row at a time
    for _, row in metadata.iterrows():
        node_id = str(row["Node ID"])
        node_folder = dataset_path / node_id

        # Check if the folder for this Node ID exists
        if not node_folder.exists():
            print(
                f"Warning: folder not found for Node ID {node_id}"
            )

            continue

        # Store JPG files found in this Node ID folder
        jpg_files = []

        for file in node_folder.iterdir():
            if file.is_file() and file.suffix.lower() == ".jpg":
                jpg_files.append(file)

        if not jpg_files:
            print(
                f"Warning: no JPG found for Node ID {node_id}"
            )

            continue
        
        # Match each JPG with its metadata row
        for image_path in jpg_files:
            record = {
                "node_id": node_id,
                "image_path": image_path,
                "metadata": row.to_dict()
            }

            records.append(record)

    return records

# test input function
if __name__ == "__main__":
    datasets = [
        "Bates_smart_Mccutcheon",
        "YFA_cabana",
        "YFA_state_government"
    ]

    for dataset in datasets:
        records = load_dataset(dataset)
        print(f"\nDataset: {dataset}")
        print(f"Matched drawings: {len(records)}")

        if records:
            first_record = records[0]
            print(
                "Node ID:",
                first_record["node_id"]
            )

            print(
                "Image:",
                first_record["image_path"]
            )

            print(
                "Title:",
                first_record["metadata"].get("Node Title")
            )