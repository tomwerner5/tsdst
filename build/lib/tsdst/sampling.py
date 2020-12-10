import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from timeit import default_timer as dt

from .utils import print_time, updateProgBar


def downSample(data, ycol, majority=0, minority=1, bal=0.5, rs=123):
    '''
    In a low base-rate classification problem, it is sometimes advantageous to downsample
    the majority class until the classes are balanced (or close enough). This 
    is done by randomly removing a proportion of observations from only the
    majority class.
    '''
    data_min = data[data[ycol] == minority]
    data_maj = data[data[ycol] == majority]

    samp_size_train = int(len(data_min)/bal - len(data_min))
    
    np.random.seed(rs)
    idx = np.random.choice(np.arange(0, len(data_maj)), samp_size_train, replace=False)

    data_maj = data_maj.iloc[idx, :]
    
    data_sub = shuffle(pd.concat((data_min, data_maj), axis=0), random_state=rs)
    
    return data_sub


def genRandSampFromDF(data, sampSize, replace=False, random_state=None):
    '''
    Creates a Random sample of observations (rows) from a Dataframe (data). If replace=True,
    this function can be used for bootstrap samples. 
    
    data: pd.DataFrame
    sampSize: Int
    replace: bool
    random_state: Int or None
    '''
    if random_state is not None:
        np.random.seed(random_state)
    idx = list(np.random.choice(np.arange(0, data.shape[0]), sampSize, replace=replace))
    if isinstance(data, (pd.DataFrame, pd.Series)):
        rand = data.iloc[idx]
    else:
        rand = data[idx]
    return rand


def latinHypercube1D(data, sampleSize, random_state=None, shuffle_after=True,
                     sort_=True, sort_method="quicksort", sort_cols=None,
                     stratified=True, bin_placement="random", verbose=False):
    '''
    Creates a sample from a Dataframe (data). If replace=True,
    this function can be used for bootstrap samples. 
    
    data: pd.DataFrame, np.ndarray, may also work with a well structured dictionary
    sampSize: Int
    random_state: Int or None
    sort_method: string
    sort_cols: list or None
    stratified: bool
    bin_placement: string
    verbose: bool
    '''
    t0 = None
    if verbose:
        t0 = dt()
        print_time("\nInitializing...", t0, te=dt())
    sortedData = None
    df = False
    if isinstance(data, pd.DataFrame):
        df = True
   
    if sort_:
        if verbose:
            print_time("\nSorting...", t0, te=dt())

        if df:
            sortedData = data.copy()
        else:
            sortedData = pd.DataFrame(data)

        if sort_cols is not None:
            sortedData = sortedData.sort_values(sort_cols, axis=0, kind=sort_method)
        else:
            sortedData = sortedData.sort_values(list(sortedData.columns), axis=0, kind=sort_method)
    
    if random_state is not None:
        np.random.seed(random_state)
        
    if verbose:
        print_time("\nShaping...", t0, te=dt())

    if sortedData is not None:
        sortedData = np.array(sortedData).reshape(data.shape[0], -1)
    else:
        sortedData = np.array(data).reshape(data.shape[0], -1)
    rows = sortedData.shape[0]
    cols = sortedData.shape[1]
    LHC = np.zeros(shape=(sampleSize, cols), dtype=data.dtypes)
    splits = sampleSize
    
    if verbose:
        print_time("\nCreating the bins...", t0, te=dt())
    
    if stratified:
        high = int(np.ceil(rows/sampleSize))
        low = int(np.floor(rows/sampleSize))
        rem = rows % sampleSize
        f_array = np.zeros(sampleSize)
        if rem != 0:
            rem2 = sampleSize - rem
            if bin_placement == "random":
                f_array = np.repeat((high, low), (rem, rem2))
                np.random.shuffle(f_array)
            elif bin_placement == "spaced":
                if rem > rem2:
                    r1 = np.repeat(high, rem)
                    a1 = np.arange(0, rem, np.floor(rem / rem2),
                                   dtype=np.int)[:rem2]
                    f_array = np.insert(r1, a1, low)
                else:
                    r1 = np.repeat(low, rem2)
                    a1 = np.arange(0, rem2, np.floor(rem2 / rem),
                                   dtype=np.int)[:rem]
                    f_array = np.insert(r1, a1, high)
            else:
                raise ValueError("""Not a valid bin placement. Change to 'spaced' or
                                 'random'. To order the bins from high to low, change
                                 stratified to False""")
        else:
            f_array = np.repeat(high, sampleSize)
        splits = np.cumsum(f_array)[:-1]
    Splits = np.array_split(sortedData, splits)
    nSplits = len(Splits)
    
    if verbose:
        print_time("\nSampling...", t0, te=dt())
       
    t1 = dt()
    for i, sample in enumerate(Splits):
        LHC[i, :] = sample[np.random.choice(sample.shape[0], 1)]
        if verbose:
            updateProgBar(i + 1, nSplits, t1)
    
    if shuffle_after:
        if verbose:
            print_time("\nShuffling...", t0, te=dt())
        np.random.shuffle(LHC)
    
    if df:
        if verbose:
            print_time("\nConverting to DataFrame...", t0, te=dt())
        LHC = pd.DataFrame(LHC, columns=data.columns).astype(data.dtypes)
        
    if verbose:
        print_time("\nFinished...", t0, te=dt())
    return LHC
