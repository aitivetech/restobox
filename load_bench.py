import time

from restobox.data.image_dataset import ImageFolderDataset


def load_files(path: str,limit: int,name:str) -> float:
    start = time.time()

    dataset = ImageFolderDataset(path)

    for i in range(limit):
        _ = dataset[i]

    end = time.time()

    total_time = end - start

    print(f"{name}: {total_time}, {total_time / limit}")
    return total_time

if __name__ == "__main__":
    load_files("/run/media/bglueck/Data/datasets/images_512x512",10000,"jpeg")
    load_files("/run/media/bglueck/Data/datasets/images_512x512_jxl",10000,"jpeg_xl")
