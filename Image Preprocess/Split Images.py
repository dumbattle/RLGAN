from PIL import Image
import os

# pkmn-regular-only.png
def main():

    while True:
        src_path = input("Please enter path to source image: ")

        exists = os.path.exists(src_path)

        if not exists:
            src_path = "../data/Source Images/" + src_path
            exists = os.path.exists(src_path)

        if exists:
            break
        print("That is not a valid path. Try again")

    src_img = Image.open(src_path)

    w = int(input("How many columns (x): "))
    h = int(input("How many rows (y): "))

    (sx, sy) = src_img.size

    ws = int(sx / w)
    hs = int(sy / h)

    prefix = input("How do you want to name these images: ")

    i = 1
    for y in range(h):
        for x in range(w):
            img = src_img.crop((ws * x, hs * y, ws * (x + 1), hs * (y + 1)))
            img.save(os.getcwd() + "/" + prefix + str(i) + ".png", "png")
            i += 1


if __name__ == "__main__":
    main()
