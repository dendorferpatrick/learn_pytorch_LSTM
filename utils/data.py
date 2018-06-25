import numpy as np


def sine_2(X, signal_freq=60.):

    return (np.sin(2 * np.pi * (X) / signal_freq) + np.sin(4 * np.pi * (X) / signal_freq)) / 2.0

def noisy(Y, noise_range=(-0.05, 0.05)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
    return Y + noise

def generate_data(data, sample_size):
    if data=='sine':    
        random_offset = np.random.randint(0, sample_size)
        X = np.arange(sample_size)
        Y = noisy(sine_2(X + random_offset)).astype('float32')
        return Y
    if data=='passenger': 
        X=np.genfromtxt("data/international-airline-passengers.csv", skip_footer= 1, usecols=1, delimiter=',', skip_header=1) 
        X=X.astype('float32') 
        return X
    if data=='stock':
        X=np.genfromtxt("data/all_stocks_5yr.csv", usecols=1, delimiter=',', skip_header=1)
        nans=np.isnan(X)
        index= np.where(nans==True)[0]
        for i in np.flip(index, 0):
            X=np.delete(X, i) 
        X=X.astype('float32')
        
        return X
    else: 
        print("No valid dataset")

def create_dataset(dataset, look_back, look_forward, normalize=True):
    if normalize:
        # Normalize data to [0, 1]
        max_value = np.max(dataset)
        min_value = np.min(dataset)   
        scalar = max_value - min_value
        dataset = list(map(lambda x: (x-min_value) / scalar, dataset))
    dataX, dataY=[], []
    for i in range(len(dataset) - look_back- look_forward):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back:(i + look_back+look_forward)])
    return np.array(dataX), np.array(dataY)

