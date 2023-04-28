import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Keypoints descriptors distance computing script')

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
    '--threshold', type=int, default=1,
    help='the maximum distance between two descriptors to be considered as a pair'
)

parser.add_argument(
    '--desc_count', type=int, default=2,
    help='the number of descriptors to keep for each point'
)

args = parser.parse_args()

def distance(desc1, desc2):
    sum = 0
    for i in range(len(desc1)):
        sum += abs(desc1[i] - desc2[i])
    return sum

descriptor_count = args.desc_count
threshold = args.threshold

desc1 = np.load(args.origin_image_file)['descriptors']
desc2 = np.load(args.destination_image_file)['descriptors']
pts1 = np.load(args.origin_image_file)['keypoints']
pts2 = np.load(args.destination_image_file)['keypoints']
transform_matrix = np.loadtxt(args.matrix_file)

# On récupère les k descripteurs de l'image 2 les plus proches de chaque descripteur de l'image 1
best_dist = []
for index1 in range(len(desc1)):
    sorted_dist = []
    distances_list = []
    for index2 in range(len(desc2)):
        distances_list.append([index2, distance(desc1[index1], desc2[index2])])
    dist = np.array(distances_list)
    sorted_indices = np.argsort(dist[:, 1])
    sorted_dist = dist[sorted_indices]
    best_dist.append(sorted_dist[:descriptor_count]) # On garde les k descripteurs les plus proches


# Pour chaque point de l'image 1 on regarde s'il correspond à un des k points de l'image 2 dont on a conservé les indices
correct = 0
booleans = [0] * len(pts1)
for i in range(len(pts1)):

    # On applique la transformation au point de l'image 1
    p = pts1[i]
    p = np.append(p, 1)
    new_p = np.matmul(p, transform_matrix)
    new_p = new_p[:-1]

    # On regarde si le point correspond à un des k points de l'image 2 retenus
    for p2 in best_dist[i]:
        index = int(p2[0])
        if np.linalg.norm(new_p - pts2[index]) <= threshold:
            correct += 1
            break

print("Correct rate: " + str(round(correct/len(pts1)*100, 2)) + "%")