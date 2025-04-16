import os

def read_annotation_file(file_path):
    """Read an annotation file and return a list of image filenames."""
    filenames = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Split on tab and take the first part as the filename
                parts = line.strip().split('\t')
                if parts:
                    filenames.append(parts[0])
    except FileNotFoundError:
        print(f"Error: Annotation file '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
    return filenames

def check_images_in_folder(annotation_files, image_folder):
    """Check if image filenames from annotation files exist in the image folder."""
    # Read filenames from both annotation files
    filenames = []
    for anno_file in annotation_files:
        file_filenames = read_annotation_file(anno_file)
        filenames.extend(file_filenames)
        print(f"Read {len(file_filenames)} lines from '{anno_file}'")

    # Get total unique filenames
    unique_filenames = set(filenames)
    print(f"\nTotal unique image filenames in annotation files: {len(unique_filenames)}")

    # Get list of images in the folder
    try:
        folder_images = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        print(f"Total images in folder '{image_folder}': {len(folder_images)}")
    except FileNotFoundError:
        print(f"Error: Image folder '{image_folder}' not found.")
        return
    except Exception as e:
        print(f"Error accessing folder '{image_folder}': {e}")
        return

    # Check for missing files
    missing_files = [fname for fname in unique_filenames if fname not in folder_images]
    
    # Output results
    if missing_files:
        print(f"\nFound {len(missing_files)} missing files:")
        for fname in missing_files:
            print(f" - {fname}")
    else:
        print("\nAll image filenames from annotation files exist in the folder.")

def main():
    # Specify the paths to the annotation files and image folder
    annotation_files = [
        'train.txt',  # Replace with the path to your first annotation file
        'val.txt'   # Replace with the path to your second annotation file
    ]
    image_folder = 'images'      # Replace with the path to your image folder

    print("Checking image filenames in annotation files against image folder...")
    check_images_in_folder(annotation_files, image_folder)

if __name__ == "__main__":
    main()