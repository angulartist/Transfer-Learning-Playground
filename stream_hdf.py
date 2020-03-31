import h5py as h5


def build_h5_stream(src, *, X_key, y_key):
    with h5.File(src, 'r') as source:
        columns = [f'feat_{feature_idx}'
                   for feature_idx
                   in range(source[X_key].shape[1])]

    def streamer():
        cursor = 0

        while True:
            with h5.File(src, 'r') as source: 
                try:
                    X_train = source[X_key][cursor]
                    y_train = source[y_key][cursor]
                except (ValueError, IndexError):
                    raise StopIteration('No more observations.')

            cursor += 1

            yield dict(zip(columns, X_train)), y_train
            
    return streamer()


stream = build_h5_stream(
    'features_train.h5',
    X_key='images',
    y_key='labels',
)

for a, b in stream:
    print(b)