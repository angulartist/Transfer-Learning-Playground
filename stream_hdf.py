import h5py as h5

def dictify(x):
    return dict(zip(['feat'] * x.shape[0], x))

def streamer():
    index = 0
    
    while True:
        with h5.File('train.h5', 'r') as file:     
            start, end = index, index + 1
            X_train = file['images'][start : end]
            y_train = file['labels'][start : end]

        index += 1

        yield X_train, y_train
        
        
gen = streamer()

a, b = next(gen)

print(a.shape)