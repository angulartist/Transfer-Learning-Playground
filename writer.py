import os 

import h5py as h5
from imutils import paths

from keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


class FileAlreadyOpenError(RuntimeError):
    pass


class HDF5ImageWriter(object):
    def __init__(self, src, dims, X_key="images", y_key="labels", buffer_size=512):

        self.src: str = src
        self.dims = dims
        self.X_key: str = X_key
        self.y_key: str = y_key
        self.db = None
        self.images = None
        self.labels = None
        self.buffer_size = buffer_size
        self.buffer = {"tmp_images": [], "tmp_labels": []}
        self._index = 0

    def __enter__(self):
        if self.db is not None:
            raise FileAlreadyOpenError("The HDF5 file is already open!")

        self.db = h5.File(self.src, "w")
        self.images = self.db.create_dataset(self.X_key, self.dims, dtype="float32")
        self.labels = self.db.create_dataset(self.y_key, (self.dims[0],), dtype="uint8")

        return self

    def __exit__(self, type_, value, traceback):
        self.__close()

    def add(self, images, labels):
        self.buffer["tmp_images"].extend(images)
        self.buffer["tmp_labels"].extend(labels)

        if len(self.buffer["tmp_images"]) >= self.buffer_size:
            self.__flush()
            
    def add_classes(self, classes):
        datatype = h5.string_dtype(encoding="utf-8")
        classes_set = self.db.create_dataset("classes", (len(classes),), dtype=datatype)
        classes_set[:] = classes
        
        print('[Classes] Added', (len(classes)))

    def __flush(self):
        index = self._index + len(self.buffer["tmp_images"])
        self.images[self._index : index] = self.buffer["tmp_images"]
        self.labels[self._index : index] = self.buffer["tmp_labels"]
        self._index = index

        self.buffer = {"tmp_images": [], "tmp_labels": []}

    def __close(self):
        if len(self.buffer["tmp_images"]) > 0:
            self.__flush()

        self.db.close()
        

        
directory = os.path.join('datasets/hotdogs/train')

X_paths = shuffle(list(paths.list_images(directory)))

classes = [path.split(os.path.sep)[-2].split('.')[0] for path in X_paths]

enc = LabelEncoder()
y = enc.fit_transform(classes)

        
#X_train, X_test, y_train, y_test = train_test_split(
#    X_paths, y, test_size=0.2, random_state=42
#)

h5_writer = HDF5ImageWriter(
    src="train.h5", dims=(len(X_paths), 299, 299, 3)
)

with h5_writer as writer:
    for path, label in zip(X_paths, y):
        raw_image = load_img(path, target_size=(299, 299))
        image = img_to_array(raw_image)
        writer.add([image], [label])
        print('Added', path, label)
    writer.add_classes(enc.classes_)