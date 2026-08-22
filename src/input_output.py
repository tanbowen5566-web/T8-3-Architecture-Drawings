from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

# Define input function
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
                "node_id": node_id,             # Store the Node ID
                "image_name": image_path.name,  # Store the JPG image name
                "image_path": image_path,       # Store the path of the JPG image

                # Convert the current metadata row to a dictionary and store it
                "metadata": row.to_dict()
            }

            records.append(record)

    return records

# Provide one image record at a time to Core
def iter_images(dataset_name):
    records = load_dataset(dataset_name)

    for record in records:
        yield record



# test input function
if __name__ == "__main__":
    datasets = [
        "Bates_smart_Mccutcheon",
        "YFA_cabana",
        "YFA_state_government"
    ]

    for dataset in datasets:
        # Load all input records from this dataset
        records = load_dataset(dataset)

        print(f"\nDataset: {dataset}")
        print(f"Matched drawings: {len(records)}")

        # Test the first record
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

        # Store images for each Node ID
        node_images = {}

        for record in records:
            node_id = record["node_id"]

            if node_id not in node_images:
                node_images[node_id] = []

            node_images[node_id].append(
                record["image_path"]
            )

        # Find Node IDs that have multiple images
        print("\nNode IDs with multiple images:")

        for node_id, images in node_images.items():
            if len(images) > 1:
                print(f"\nNode ID: {node_id}")
                print(
                    f"Number of images: {len(images)}"
                )

                for image in images:
                    print(" -", image)

    # Test iter_images function
    print("\n--- Testing iter_images ---")

    count = 0

    for record in iter_images("Bates_smart_Mccutcheon"):
        print("\nImage received by Core:")
        print("Node ID:", record["node_id"])
        print("Image name:", record["image_name"])
        print("Image path:", record["image_path"])

        count += 1

        # Only test the first 3 images
        if count == 3:
            break


# Define output function
def save_output(results, output_path):
    # Check whether there are results to save
    if not results:
        print("Warning: no output results to save")
        return

    # Convert the core results into a table
    output_df = pd.DataFrame(results)

    # Define the required output columns
    output_columns = [
        "project_name",
        "drawing_name",
        "architect",
        "draughtsperson",
        "drawing_number",
        "date",
        "scale",
        "drawing_types"
    ]

    # Add missing columns as empty values
    for column in output_columns:
        if column not in output_df.columns:
            output_df[column] = ""
    
    # Convert multiple drawing types into one string for CSV
    if "drawing_types" in output_df.columns:
        output_df["drawing_types"] = output_df["drawing_types"].apply(
            lambda x: "; ".join(x) if isinstance(x, list) else x
        )

    # Keep the columns in the required order
    output_df = output_df[output_columns]

    # Convert output_path into a Path object
    output_path = Path(output_path)

    # Create the output folder if it does not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the results to a CSV file
    output_df.to_csv(output_path, index=False)

    print(f"Output saved to: {output_path}")


#Test output function
test_results = [
    {
        "project_name": "Test Project",
        "drawing_name": "Site Plan",
        "architect": "Test Architect",
        "draughtsperson": "Test Person",
        "drawing_number": "001",
        "date": "1984",
        "scale": "1:100",
        "drawing_types": [
            "Front View",
            "Left View",
            "Right View"
        ]
    }
]

save_output(test_results, "output/extracted_metadata.csv")