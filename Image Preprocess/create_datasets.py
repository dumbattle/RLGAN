import os
from PIL import Image
import numpy as np


def main():
    while True:
        src_path = input("Please enter folder path to source images: ")

        exists = os.path.isdir(src_path)

        if not exists:
            src_path = "../data/" + src_path
            exists = os.path.exists(src_path)

        if exists:
            break
        print("That is not a valid path. Try again")

    files = os.listdir(src_path)

    count = 0
    result = []
    for f in files:
        if f[-4:] != ".png":
            continue
        img = Image.open(src_path + "/" + f)
        a = np.array(img)
        result.append(a)
        count += 1
    result = np.stack(result)
    print(f"The shape of the dataset is: {result.shape}")
    ss = input("Where would you like to save this: ")
    np.save(ss, result)


if __name__ == "__main__":
    main()
