import numpy as np  
import argparse

parser = argparse.ArgumentParser(description='Keypoints distance computing script')

parser.add_argument(
    '--origin_image_file', type=str, required=True,
    help='path to a file containing the keypoints and descriptors of the first image'
)

parser.add_argument(
    '--destination_image_file', type=str, required=True,
    help='path to a file containing the keypoints and descriptors of the second image'
)

parser.add_argument(
    '--matrix_file', type=str, required=True,
    help='path to a file containing the transform matrix from the origin image to the destination image'
)

parser.add_argument(
    '--threshold', type=str, default=1,
    help='the maximum distance between two points to be considered as a pair'
)

args = parser.parse_args()

threshold = args.threshold

origin_arrays = np.load(args.origin_image_file)
origin_keypoints = origin_arrays['keypoints']

destination_arrays = np.load(args.destination_image_file)
destination_keypoints = destination_arrays['keypoints']

transform_matrix = np.loadtxt(args.matrix_file)

origin_keypoints_transform = np.hstack(np.array([0,0]))
keypoints_distances = np.hstack(np.array([0 for _ in range(destination_keypoints.shape[0])]))

print("----")
print("Counting keypoints pairs between", args.origin_image_file, "and", args.destination_image_file)
print("Threshold is set to", args.threshold)
print(f"{origin_keypoints.shape[0]} keypoints in {args.origin_image_file}")
print(f"{destination_keypoints.shape[0]} keypoints in {args.destination_image_file}")
print("----")

for og_keypoint in origin_keypoints:
    og_keypoint = np.append(og_keypoint, 1)
    keypoint_transform = np.matmul(og_keypoint, transform_matrix)
    keypoint_transform = keypoint_transform[:-1]

    keypoint_distance_vector = np.empty(0)
    for dst_keypoint in destination_keypoints:
        keypoint_distance_vector = np.append(keypoint_distance_vector, np.linalg.norm(keypoint_transform-dst_keypoint))

    keypoints_distances = np.vstack((keypoints_distances, keypoint_distance_vector))
    origin_keypoints_transform = np.vstack((origin_keypoints_transform, keypoint_transform))

keypoints_distances = np.delete(keypoints_distances, 0, 0)
valid_pairs_count = 0

for distances in keypoints_distances:
    if np.amin(distances) <= threshold: 
        valid_pairs_count += 1

print("Done! Found", valid_pairs_count, "valid pairs, which is", (valid_pairs_count / max(origin_keypoints.shape[0], destination_keypoints.shape[0]))*100, "% of the possible maximum")