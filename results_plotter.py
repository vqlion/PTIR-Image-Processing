import numpy as np
import matplotlib.pyplot as plt
import json

with open('message.json', 'r') as file:
    json_file = json.load(file)

for dataset in json_file:
    fig, axs = plt.subplots(2, 5, sharex=True, figsize=(18, 8))
    title = ""
    path = dataset[0]["image_1"].split("\\")[:-1]
    if len(path) == 0:
        path = dataset[0]["image_1"].split("/")[:-1]
    for i in path:
        title += i + "/"
    fig.suptitle(title)

    for j in range(len(dataset[0]) + 1):
        for i in range(2):
            axs[i][j].set_xlabel('Threshold')
            axs[i][j].set_ylabel('Success rate')
            axs[i][j].set_xlim(0, 50)
            axs[i][j].set_ylim(0, 100)
            axs[i][j].grid(True)

        axs[0][j].plot(dataset[j]["keypoints_distances"])
        axs[0][j].set_title("Keypoints")
        axs[1][j].plot(dataset[j]["descriptors_distances"])
        axs[1][j].set_title("Descriptors")

    plt.show()
