from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import os


def download(data_path='./'):
    """Download official Malaria dataset to data_path"""
    url_zip_file = 'https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip'
    with urlopen(url_zip_file) as file:
        with ZipFile(BytesIO(file.read())) as zfile:
            zfile.extractall(data_path)


def rename_data_folder(data_path='./'):
    """Rename data folder structure as : ./data/1 for parasitized and ./data/0 for un-infected"""
    src = os.path.join(data_path, "cell_images")
    dst = os.path.join(data_path, "data")
    os.rename(src, dst)
    os.rename(os.path.join(dst, "Parasitized"), os.path.join(dst, "1"))
    os.rename(os.path.join(dst, "Uninfected"), os.path.join(dst, "0"))


if __name__ == "__main__":
    download()
    rename_data_folder()
