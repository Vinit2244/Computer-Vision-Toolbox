import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse

# Load the annotation JSON file
annotations_folder = "dataset/data/annots"
images_folder = "dataset/data/img"

def draw_obb(image, xc, yc, w, h, theta):
    box = ((xc, yc), (w, h), theta)
    box_points = cv2.boxPoints(box)
    box_points = np.int32(box_points)
    
    # Draw the bounding box
    cv2.drawContours(image, [box_points], 0, (255, 0, 0), 2)  # Blue bounding box

def visualise_dataset(n_rows=3, n_cols=5, width=20, height=12, save=True):
    image_files = glob.glob(os.path.join(images_folder, "*.jpg"))
    image_filenames = [os.path.basename(img) for img in image_files]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height))

    image_idx = -1
    for i in range(n_rows):
        for j in range(n_cols):
            image_idx += 1

            annotation_file = f"{annotations_folder}/{image_filenames[image_idx]}.json"
            with open(annotation_file, "r") as f:
                data = json.load(f)

            # Load the corresponding image
            image = cv2.imread(f"{images_folder}/{image_filenames[image_idx]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Iterate over objects in annotation and draw them
            for obj in data["objects"]:
                obb = obj["obb"]
                draw_obb(image, obb["xc"], obb["yc"], obb["w"], obb["h"], obb["theta"])

            # Show the image with bounding boxes
            axes[i][j].imshow(image)
            axes[i][j].axis("off")

    plt.suptitle("Oriented Bounding Boxes Visualization", fontsize=20)
    plt.tight_layout()
    if save:
        plt.savefig("dataset_visualisation.png")
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--r", type=int, default=3, help="Number of rows in the grid")
    argparser.add_argument("--c", type=int, default=5, help="Number of columns in the grid")
    argparser.add_argument("--w", type=int, default=20, help="Width of the figure")
    argparser.add_argument("--h", type=int, default=12, help="Height of the figure")
    argparser.add_argument("--save", type=bool, default=True, help="Save the visualisation as an image")
    args = argparser.parse_args()

    visualise_dataset(args.r, args.c, args.w, args.h, args.save)