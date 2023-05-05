import os
import argparse
import sys
import json

from test_descriptors_distance import descriptors_distance
from test_keypoints_distance import keypoints_distance
from score_sorter import score_sorter
from test_scores import scores

parser = argparse.ArgumentParser(description='Launch the tests')

parser.add_argument(
    '--path', type=str, required=True,
    help='path of the directory containing the npz files'
)

parser.add_argument(
    '--model', type=str, required=True,
    help='name of the model used'
)

parser.add_argument(
    '--main_file', type=str, default="1.ppm.npz",
    help='name of the main file (compared to all other npz files in directory)'
)

parser.add_argument(
    '--output_file', type=str, default="output.json",
    help='name of the output file to append the output to'
)

parser.add_argument(
    '--threshold_range', type=int, default=50,
    help='number of threshold to test'
)

parser.add_argument(
    '--number_of_points', type=int, default=50,
    help='number of keypoints to keep for each image'
)

args = parser.parse_args()
main_file_name = args.main_file
output_file = args.output_file
threshold_range = args.threshold_range
number_of_points = args.number_of_points
model = args.model

main_file = ""
path = args.path
extension = '.npz'

try:
    json_output = json.load(open(output_file))
except:
    json_output = []

for root, dirs, files in os.walk(path):
    for file in files:
        if file == main_file_name:
            main_file = file

    if not main_file:
        print("No main file was found in this directory.")
        sys.exit()
    
    absolute_main_file_name = main_file[0]
    main_file_path = os.path.join(root, main_file_name)
    score_sorter(main_file_path, number_of_points)
    scores_output_array = [0 for _ in range(threshold_range)]

    for i in range(threshold_range):
        scores_output_array[i] = scores(root, i)

    output_object = {"model": model,
                    "image_folder": root,
                    "scores": scores_output_array,
                    "images_tests": []
                    }

    for file in files:
        if file == main_file or os.path.splitext(file)[-1].lower() != extension:
            continue

        absolute_file_name = file[0]
        matrix_file = f"H_{absolute_main_file_name}_{absolute_file_name}"

        if not(matrix_file in files):
            print("No matrix file was found for", file)
            continue
        
        file_path = os.path.join(root, file)
        matrix_file_path = os.path.join(root, matrix_file)

        print("Comparing ", main_file, " and ", file, " with transform matrix ", matrix_file)

        score_sorter(file_path, number_of_points)

        keypoints_distances_output = [0 for _ in range(threshold_range)]
        descriptors_distances_output = [0 for _ in range(threshold_range)]

        for i in range(threshold_range):
            keypoints_distances_output[i] = keypoints_distance(main_file_path, file_path, matrix_file_path, i)
            descriptors_distances_output[i] = descriptors_distance(main_file_path, file_path, matrix_file_path, i, 3)

        output_object["images_tests"].append({
            "image_1": main_file_path,
            "image_2": file_path,
            "keypoints_distances": keypoints_distances_output,
            "descriptors_distances": descriptors_distances_output
        })

        # print(keypoints_distance(main_file_path, file_path, matrix_file_path, 10))
        # print(descriptors_distance(main_file_path, file_path, matrix_file_path, 10, 2))

    json_output.append(output_object)

with open(output_file, "w") as file:
    json.dump(json_output, file)
