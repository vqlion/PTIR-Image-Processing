import numpy as np  
import argparse

# parser = argparse.ArgumentParser(description='Keypoints distance computing script')

# parser.add_argument(
#     '--origin_image_file', type=str, required=False,
#     help='path to a file containing the keypoints and descriptors of the first image'
# )

# parser.add_argument(
#     '--destination_image_file', type=str, required=False,
#     help='path to a file containing the keypoints and descriptors of the second image'
# )

# parser.add_argument(
#     '--matrix_file', type=str, required=False,
#     help='path to a file containing the transform matrix from the origin image to the destination image'
# )

# parser.add_argument(
#     '--threshold', type=int, default=1,
#     help='the maximum distance between two points to be considered as a pair'
# )

# args = parser.parse_args()

def keypoints_distance(file1, file2, matrix_file, threshold):

    origin_keypoints = np.load(file1)['keypoints']
    destination_keypoints = np.load(file2)['keypoints']
    transform_matrix = np.loadtxt(matrix_file)

    origin_keypoints_transform = np.hstack(np.array([0,0]))
    keypoints_distances = np.hstack(np.array([0 for _ in range(destination_keypoints.shape[0])]))

    # print("----")
    # print("Counting keypoints pairs between", file1, "and", file2)
    # print("Threshold is set to", threshold)
    # print(f"{origin_keypoints.shape[0]} keypoints in {file1}")
    # print(f"{destination_keypoints.shape[0]} keypoints in {file2}")
    # print("----")

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

    for distances_list in keypoints_distances:
        for distance in distances_list:
            if distance <= threshold:
                valid_pairs_count += 1
                break

    return((valid_pairs_count / origin_keypoints.shape[0])*100)

# print(keypoints_distance(args.origin_image_file, args.destination_image_file, args.matrix_file, args.threshold))
