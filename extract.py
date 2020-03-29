import os 

import h5py as h5
import numpy as np

from keras.applications import inception_v3, resnet50, imagenet_utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from h5imagegenerator import HDF5ImageGenerator


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


model = resnet50.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(299, 299, 3),
)
#model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

batch_size = 32

gen = HDF5ImageGenerator(
    src= 'train.h5',
    classes_key='classes',
    num_classes=2,
    labels_encoding=False,
    scaler=False,
    batch_size=batch_size
)

h5_writer = HDF5ImageWriter(
    src="features_train.h5",
    dims=(gen.num_items, 2048 * 10 * 10),
    buffer_size=batch_size
)

with h5_writer as writer:
    for batch_X, batch_y in gen:
        batch_X = resnet50.preprocess_input(batch_X)
        features = model.predict(batch_X, batch_size=batch_size)
        features = features.reshape((features.shape[0], 2048 * 10 * 10))
        
        writer.add(features, batch_y)
        print('Added', batch_y)
    writer.add_classes(gen.classes)