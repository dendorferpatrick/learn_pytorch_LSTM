import numpy as np

np.random.seed(0)
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
        #Y=np.sin( X*2*np.pi/60. ).astype('float32')
        return Y
    if data=='linear':    
        random_offset = np.random.randint(0, 10)
        X = np.arange(sample_size)
        Y = (random_offset + X* np.random.rand()).astype('float32')
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

def create_dataset(dataset,train_size, look_back, look_forward, difference= False, normalize=True):
    # Slit data to train and test data
    datatrain= np.array(dataset[:train_size])
    datatest = np.array(dataset[train_size:])

    if normalize:
        # Normalize data to [0, 1]
        max_value = np.max(datatrain)
        min_value = np.min(datatrain)   
        scalar = max_value - min_value
        datatrain = np.array(list(map(lambda x: (x-min_value) / scalar, datatrain )))
        datatest = np.array(list(map(lambda x: (x-min_value) / scalar, datatest)))
    train_X, train_Y,val_X, val_Y =[], [], [],[]
    if difference:
        for i in range(len(datatrain) - look_back- look_forward-1):
            a = datatrain[i+1:(i+1 + look_back)]- datatrain[i:(i + look_back)]
            train_X.append(a)
            train_Y.append(datatrain[i+1 + look_back:(i+1 + look_back+look_forward)]-datatrain[i+ look_back:(i+ look_back+look_forward)])
        for i in range(len(datatest) - look_back- look_forward- 1):
            a = datatest[i+1:(i+1 + look_back)]-datatest[i:(i + look_back)]
            test_X.append(a)
            test_Y.append(datatest[i+1 + look_back:(i+1 + look_back+look_forward)]-datatest[i + look_back:(i + look_back+look_forward)]) 

    else:
        for i in range(len(datatrain) - look_back- look_forward):
            a = datatrain[i:(i + look_back)]
            train_X.append(a)
            train_Y.append(datatrain[i + 1:(i + look_back+1)])
        for i in range(len(datatest) - look_back- look_forward):
            a = datatest[i:(i + look_back)]
            val_X.append(a)
            val_Y.append(datatest[i + 1:(i + look_back+1)])
        test_X=datatest[:-1]
        test_Y=datatest[1:]
    return np.array(train_X).T, np.array(train_Y).T, np.array(test_X), np.array(test_Y), np.array(val_X).T, np.array(val_Y).T

