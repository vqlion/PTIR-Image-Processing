import os
import argparse

def loop_through_dir(path, extension):
    tmp = ''
    for root, dirs, files in os.walk(path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext == extension:
                tmp += os.path.join(root, file) + '\r\n'
    return tmp

    
parser = argparse.ArgumentParser(description='Get path to all images in directory in a txt file.')

parser.add_argument(
    '--path', type=str, required=True,
    help='path of the directory'
)

parser.add_argument(
    '--type', type=str, required=True,
    help='type of image files'
)

args = parser.parse_args()

directory = args.path
filetype = args.type

res = loop_through_dir(directory, filetype)

with open ('image_list.txt', 'w') as file:  
    file.write(res)  