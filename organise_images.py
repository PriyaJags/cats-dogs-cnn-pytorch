from pathlib import Path
import shutil

# Location of the downloaded training images
source_train = Path(
    r"C:\Users\priya\Downloads\cats_dogs_light\cats_dogs_light\train"
)

# The test folder is alongside the train folder
source_test = source_train.parent / "test"

# Destination folders inside our CNN project
project_folder = Path(__file__).resolve().parent
destination = project_folder / "cats_dogs"

# Organise training and testing images
for split, source in [
    ("train", source_train),
    ("test", source_test)
]:
    for animal in ["cat", "dog"]:

        # Create the destination class folder
        target = destination / split / animal
        target.mkdir(parents=True, exist_ok=True)

        # Find matching image files
        images = [
            file for file in source.glob(f"{animal}.*")
            if file.is_file()
        ]

        # Copy images into the correct folder
        for image_file in images:
            shutil.copy2(image_file, target / image_file.name)

        print(f"{split} {animal}: {len(images)} images copied")

print("Image organisation complete!")