import os, sys
from PIL import Image, ImageOps
import pandas as pd
from tqdm import tqdm
import os
import errno

size = 320, 320

def main():
    train_files = pd.read_csv("./config/train.csv")

    n_errors = 0
    errors = set()

    for path in tqdm(train_files["Path"]):
        try:
            im = Image.open(".{}".format(path))
            im = ImageOps.fit(im, size, Image.ANTIALIAS)

            FILEPATH = "./DenseNet-im{}".format(path)
            if not os.path.exists(os.path.dirname(FILEPATH)):
                try:
                    os.makedirs(os.path.dirname(FILEPATH))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise

            im.save(FILEPATH, "JPEG")
        except:
            n_errors += 1
            errors.add(path)

    print("Done. Encountered {} errors. Errors:".format(n_errors))
    # print(errors)

if __name__ == "__main__":
    main()
