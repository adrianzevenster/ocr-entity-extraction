import os
import zipfile
import sqlite3
import re
from collections import Counter, defaultdict

import numpy as np
import h5py
import pytesseract
from PIL import Image
import spacy
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# Step 1: Extract the zipped dataset
# ------------------------------------------------------------------------------
def extract_zip(zip_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted '{zip_path}' to '{extract_dir}'")


# ------------------------------------------------------------------------------
# Step 2: Create HDF5 database from extracted images
# ------------------------------------------------------------------------------
def create_hdf5_from_images(image_dir, hdf5_path):
    # Gather image paths
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    num_images = len(image_paths)
    if num_images == 0:
        raise ValueError("No images found in the extracted directory!")
    print(f"Found {num_images} images.")

    # Determine target image dimensions from the first image
    sample_img = Image.open(image_paths[0]).convert('L')
    target_width, target_height = sample_img.size
    print(f"Target image size: {target_width} x {target_height}")

    # Create the HDF5 file and dataset with chunked writes
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, 'w') as f:
        dset = f.create_dataset(
            'images',
            shape=(num_images, target_height, target_width),
            dtype='uint8',
            chunks=True  # Enable chunked storage so that we don't hold everything in memory at once.
        )
        # Process and write images one by one
        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert('L').resize((target_width, target_height))
                dset[i, ...] = np.array(img)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
            if (i + 1) % 100 == 0 or (i + 1) == num_images:
                print(f"Processed {i + 1}/{num_images} images")
    print(f"HDF5 database created at '{hdf5_path}'")



# ------------------------------------------------------------------------------
# Step 3: Perform OCR on HDF5 images and store results in SQLite
# ------------------------------------------------------------------------------
def perform_ocr_on_hdf5(hdf5_path, sqlite_db_path):
    with h5py.File(hdf5_path, 'r') as f:
        images = f['images'][:]
    num_images = images.shape[0]
    print(f"Performing OCR on {num_images} images...")

    os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ocr_results (
            image_id INTEGER PRIMARY KEY,
            ocr_text TEXT
        )
    ''')
    conn.commit()

    for i in range(num_images):
        pil_image = Image.fromarray(images[i])
        ocr_text = pytesseract.image_to_string(pil_image)
        cursor.execute('''
            INSERT OR REPLACE INTO ocr_results (image_id, ocr_text)
            VALUES (?, ?)
        ''', (i, ocr_text))
        print(f"OCR processed for image {i}")
    conn.commit()
    conn.close()
    print("OCR extraction completed and stored in SQLite database.")


# ------------------------------------------------------------------------------
# Step 4: Perform entity extraction on OCR results and visualize the results
# ------------------------------------------------------------------------------
def entity_extraction_and_visualization(sqlite_db_path):
    nlp = spacy.load("en_core_web_sm")

    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_extraction (
            image_id INTEGER PRIMARY KEY,
            entities TEXT
        )
    ''')
    conn.commit()

    cursor.execute("SELECT image_id, ocr_text FROM ocr_results")
    rows = cursor.fetchall()

    label_counts = Counter()
    entity_counts = defaultdict(Counter)

    for image_id, ocr_text in rows:
        if not ocr_text:
            continue
        clean_text = ocr_text.lower().strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = re.sub(r'[^\x00-\x7F]+', ' ', clean_text)
        doc = nlp(clean_text)
        extracted_entities = "; ".join([f"{ent.text}:{ent.label_}" for ent in doc.ents])
        cursor.execute('''
            INSERT OR REPLACE INTO entity_extraction (image_id, entities)
            VALUES (?, ?)
        ''', (image_id, extracted_entities))
        for ent in doc.ents:
            label_counts[ent.label_] += 1
            entity_counts[ent.label_][ent.text] += 1
        print(f"Entities extracted for image {image_id}: {extracted_entities}")

    conn.commit()
    conn.close()

    labels = list(label_counts.keys())
    counts = [label_counts[label] for label in labels]
    plt.figure(figsize=(12, 6))
    plt.barh(labels, counts)
    plt.xlabel('Frequency')
    plt.ylabel('Entity Label')
    plt.title('Frequency of Entity Labels')
    plt.tight_layout()
    plt.show()

    for label, counter in entity_counts.items():
        print(f"Top entities for '{label}':")
        for entity, cnt in counter.most_common(5):
            print(f"  {entity}: {cnt}")


# ------------------------------------------------------------------------------
# Main pipeline: execute all steps in order
# ------------------------------------------------------------------------------
def main():
    base_dir = os.getcwd()
    zip_path = os.path.join(base_dir, "Dataset", "BID Dataset.zip")
    extract_dir = os.path.join(base_dir, "Datasets", "HDF5", "extracted_images")
    hdf5_path = os.path.join(base_dir, "Datasets", "HDF5", "bid_dataset.HDF5")
    sqlite_db_path = os.path.join(base_dir, "Datasets", "SQLite", "ocr_results.db")

    print("=== Step 1: Extract zipped dataset ===")
    extract_zip(zip_path, extract_dir)

    print("\n=== Step 2: Create HDF5 database from images ===")
    create_hdf5_from_images(extract_dir, hdf5_path)

    print("\n=== Step 3: Perform OCR on images and store results ===")
    perform_ocr_on_hdf5(hdf5_path, sqlite_db_path)

    print("\n=== Step 4: Perform entity extraction and visualize results ===")
    entity_extraction_and_visualization(sqlite_db_path)

    print("\nPipeline execution completed successfully.")


if __name__ == "__main__":
    main()
