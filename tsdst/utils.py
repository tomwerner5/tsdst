
import datetime
import os
import pickle
import random
import sys
from bisect import bisect_right
from getpass import getpass
from timeit import default_timer as dt

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url


def dsn_getTable(sql, user, dialect, dsn, pw=None, create_engine_args=None, read_sql_args=None):
    r"""Load data into python from a SQL database.

    Requires a DSN or connection string. Uses SQLAlchemy to make the
    connection, and closes the connection on completion.
    See https://docs.sqlalchemy.org/en/20/core/engines.html for
    help building connection strings.

    Parameters
    ----------
    sql : str, or list
        A string containing the SQL query to run (It should be something
        that returns a table), or a list of sql statements to return multiple
        tables (tables are returned as a list of pandas dataframes, in this
        case).
    user : str
        The username for the database application, if applicable.
    dialect : str
        The database flavor (oracle, mysql, etc.)
    dsn : str
        Can either be a saved DSN from your computer, or a string that
        represents all the relevant information provided in a DSN.
        It can also be a host:port combination. See sqlalchemy
        documentation for more details.
    pw : str, optional
        The password for the database application that matches the user
        specified, if applicable. If None or '', getpass will ask for a
        password. If there is no password, just leave it blank.
        The default is None.
    create_engine_args : dict, optional
        A dictionary containing arguments to be passed to SQLAlchemy's
        create_engine function.
    read_sql_args : dict, optional
        A dictionary containing arguments to be passed to pandas'
        read_sql function.

    Returns
    -------
    data : pandas dataframe or list
        A dataframe (or list of dataframes) containing the returned data.

    Notes
    -----
    The DSN may contain a host, port, database path, and query parameters.
    Those components are parsed before credentials are assigned, so characters
    such as @, /, and : in passwords are escaped correctly. The engine is
    disposed even when a query or write raises an exception. These helpers
    build credentialed DSN URLs; SQLite uses a different URL form and can be
    used directly through a SQLAlchemy engine and pandas' SQL functions.
    """
    if not pw:
        pw = getpass(prompt='Password: ')
    url = make_url(dialect + "://" + dsn).set(username=user, password=pw)
    engine = create_engine(url, **(create_engine_args or {}))
    try:
        if isinstance(sql, str):
            return pd.read_sql(sql, engine, **(read_sql_args or {}))
        return [pd.read_sql(query, engine, **(read_sql_args or {})) for query in sql]
    finally:
        engine.dispose()


def dsn_saveTable(data, tableName, user, dialect, dsn, pw=None, create_engine_args=None, to_sql_args=None):
    r"""Save pandas dataframe to database.

    Parameters
    ----------
    data : pandas dataframe
        The data to save to a table.
    tableName : str
        The name of the table in the database.
    user : str
        The username for the database application, if applicable.
    dialect : str
        The database flavor (oracle, mysql, etc.)
    dsn : str
        Can either be a saved DSN from your computer, or a string that
        represents all the relevant information provided in a DSN.
        It can also be a host:port combination. See sqlalchemy
        documentation for more details.
    pw : str, optional
        The password for the database application that matches the user
        specified, if applicable. If None or '', getpass will ask for a
        password. If there is no password, just leave it blank.
        The default is None.
    create_engine_args : dict, optional
        A dictionary containing arguments to be passed to SQLAlchemy's
        create_engine function. The default is None.
    to_sql_args : dict, optional
        A dictionary containing arguments to be passed to pandas'
        to_sql function. The default is None.

    Returns
    -------
    None.

    Notes
    -----
    The DSN may contain a host, port, database path, and query parameters.
    Those components are parsed before credentials are assigned, so characters
    such as @, /, and : in passwords are escaped correctly. The engine is
    disposed even when a query or write raises an exception. These helpers
    build credentialed DSN URLs; SQLite uses a different URL form and can be
    used directly through a SQLAlchemy engine and pandas' SQL functions.
    """
    if not pw:
        pw = getpass(prompt='Password: ')
    url = make_url(dialect + "://" + dsn).set(username=user, password=pw)
    engine = create_engine(url, **(create_engine_args or {}))
    try:
        data.to_sql(tableName, engine, **(to_sql_args or {}))
    finally:
        engine.dispose()


def pretty_print_time(ts, te=None, decimals=0):
    """
    Convert time in seconds to a clock-like time, i.e. 00:00:00.00 format.

    Parameters
    ----------
    ts : float
        Start time (or elapsed time if te is None).
    te : float, optional
        End time. The default is None.
    decimals : int, optional
        The number of decimal places to use for tracking miliseconds.
        The default is 0

    Returns
    -------
    pretty : str
        Prettified time.

    """
    if te is None:
        t = ts
    else:
        t = te - ts
    
    hours_placeholder = t/3600.0
    hours = np.floor(hours_placeholder)
    
    minutes_placeholder = (hours_placeholder - hours) * 60.0
    minutes = np.floor(minutes_placeholder)
    
    seconds = np.round((minutes_placeholder - minutes) * 60.0, decimals)
    
    h = str(int(hours))
    if len(h) == 1:
        h = "0" + h
    
    m = str(int(minutes))
    if len(m) == 1:
        m = "0" + m
    
    s = str(int(seconds))
    if len(s) == 1:
        s = "0" + s
    
    if decimals != 0:
        second_decimal = seconds - int(seconds)
        s = '.'.join([s, str(int(second_decimal*10**decimals))])
    
    pretty = h + ":" + m + ":" + s
    return pretty


def checkDir(directory, make=True, verbose=True):
    """
    Check if a directory exists. If it doesn't, create it.

    Parameters
    ----------
    directory : str
        Directory to check.
    make : bool, optional
        Whether or not to create a missing directory. The default is True.
    verbose : bool
        Print results.

    Returns
    -------
    bool
        True if the directory exists.

    """
    found = False
    msg = ''
    if not os.path.isdir(directory):
        msg = msg + 'Directory not found. '
        if make:
            os.mkdir(directory)
            msg = msg + 'Directory ' + directory + ' created.'
        else:
            msg = msg + 'Directory ' + directory + 'not created.'
    else:
        found = True
    
    if verbose:
        print(msg)
    return found


def print_message_with_time(msg, ts, te=None, display_realtime=True,
                            backsn=False, log=False, log_dir="log",
                            log_filename="pmwt.log", log_args="a",
                            time_first=False, decimals=0):
    """
    Print a message with a timestamp. 

    Parameters
    ----------
    msg : str
        The message you want to print.
    ts : float
        Start time in seconds, or elapsed time if te is None.
    te : float, optional
        End time in seconds. The default is None.
    display_realtime : bool, optional
        Display the system (calendar) time as part of the output.
        The default is True.
    backsn : bool, optional
        Add '\\n'. The default is False.
    log : bool, optional
        Save message to a log file. The default is False.
    log_dir : str, optional
        The directory for the log file. The default is "log".
    log_filename : str, optional
        The name of the logfile. The default is "pmwt.log".
    log_args : str, optional
        The read/write specification, for example, wb, w, a, etc.
        The default is "a" for append. See open function in python for more
        details.
    time_first : bool, optional
        Place time at the begininng of the message. The default is False.
    decimals : int, optional
        The number of decimal places to use for tracking miliseconds.
        The default is 0

    Returns
    -------
    printed : str
        Printed message with time.

    """
    date_time = datetime.datetime.now().strftime("%I:%M:%S %p (%b %d)")
    if display_realtime:
        time_str = pretty_print_time(ts, te, decimals) + " Current Time: " + date_time
    else:
        time_str = pretty_print_time(ts, te, decimals)
    if time_first:
        printed = time_str + msg
    else:
        printed = msg + time_str
    if backsn:
        printed = printed + "\n"
    sys.stdout.write(printed)
    sys.stdout.flush()
    if log:
        checkDir(log_dir)
    logfile = log_dir + "/" + log_filename
    logfile = logfile.replace("//", "/")
    if log:
        with open(logfile, log_args) as f:
            f.write(printed)
    return printed
    

def print_time(*args, **kwargs):
    """
    Wrapper for print_message_with_time. Shortened for ease of use.

    Parameters
    ----------
    *args : 
        Positional arguments (passed to print_message_with_time).
    **kwargs : 
        Keyword Arguments (passed to print_message_with_time).

    Returns
    -------
    None.

    """
    print_message_with_time(*args, **kwargs)


def save_checkpoint(obj, msg, ts=None, te=None, decimals=0,
                    display_realtime=True, backsn=False,
                    log=True, log_dir="log", log_filename="", log_args="a",
                    time_first=False, checkpoint_dir="checkpoints",
                    checkpoint_filename="chkpnt", checkpoint_extension="pkl",
                    checkpoint_args="wb"):
    """
    Save a python object to a pickle file as a checkpoint. Log message as 
    logfile with the export.

    Parameters
    ----------
    obj : any python object (must be pickleable)
        The object to pickle.
    msg : str
        The message you want to print.
    ts : float
        Start time in seconds, or elapsed time if te is None.
    te : float, optional
        End time in seconds. The default is None.
    decimals : int, optional
        The number of decimal places to use for tracking miliseconds.
        The default is 0
    display_realtime : bool, optional
        Display the system (calendar) time as part of the output.
        The default is True.
    backsn : bool, optional
        Add '\\n'. The default is False.
    log : bool, optional
        Save message to a log file. The default is False.
    log_dir : str, optional
        The directory for the log file. The default is "log".
    log_filename : str, optional
        The name of the logfile. The default is "pmwt.log".
    log_args : str, optional
        The read/write specification, for example, wb, w, a, etc.
        The default is "a" for append. See open function in python for more
        details.
    time_first : bool, optional
        Place time at the begininng of the message. The default is False.
    checkpoint_dir : str, optional
        The directory to save the checkpoint. The default is "checkpoints".
    checkpoint_filename : str, optional
        The filename for the checkpoint. The default is "chkpnt".
    checkpoint_extension : str, optional
        The file extension for the checkpoint. The default is "pkl".
    checkpoint_args : str, optional
        The read/write specification, for example, wb, w, a, etc.
        See open function in python for more details.
        The default is "wb" for write binary.

    Returns
    -------
    None.

    """
    print_message_with_time(msg=msg, ts=ts, te=te, decimals=decimals,
                            display_realtime=display_realtime,
                            backsn=backsn, log=log, log_dir=log_dir,
                            log_filename=log_filename,
                            log_args=log_args, time_first=time_first)
    checkDir(checkpoint_dir)
    file = checkpoint_dir + '/' + checkpoint_filename + "." + checkpoint_extension
    file = file.replace('//', '/')
    if checkpoint_extension == 'pkl':
        with open(file, checkpoint_args) as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def updateProgBar(curIter, totalIter, t0, barLength=20, decimals=0):
    """
    Update progress bar. Place this function anywhere in a loop where you want
    to keep track of the loop's progress.

    Parameters
    ----------
    curIter : int
        The current iteration.
    totalIter : int
        The total number of iterations. 
    t0 : numeric
        The start time of the operation (in seconds).
    barLength : int, optional
        The length of the progress bar. The default is 20.
    decimals : int, optional
        The number of decimal places to use for tracking miliseconds.
        The default is 0

    Returns
    -------
    None.

    """
    status = "Working..."
    progress = float(curIter)/float(totalIter)
    if isinstance(progress, int):
        progress = float(progress)
    if progress >= 1:
        progress = 1
        status = "Finished!..."
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.2f}% iter: {2}/{3} {4} Elapsed: {5}, Estimated: {6}".format(
        "#"*block + "-"*(barLength - block), 
        round(progress*100.0, 2), curIter, totalIter, status,
        pretty_print_time(t0, dt(), decimals=decimals),
        pretty_print_time((dt()-t0)/curIter * (totalIter - curIter),
                          decimals=decimals))
    if progress >= 1:
        sys.stdout.write(text + "\r\n")
        sys.stdout.flush()
    else:
        sys.stdout.write(text)
        sys.stdout.flush()
        
        
def move_columns_to_end(data, col):
    """
    Rearrange a pandas dataframe by moving a column (or list of columns)
    to the end of the dataframe.

    Parameters
    ----------
    data : pandas dataframe
        Data to rearrange.
    col : str or list
        String containing one column to move, or list containing several.

    Returns
    -------
    data : pandas dataframe
        Data with rearranged columns.

    """
    if isinstance(col, str):
        col = [col]
    else:
        col = list(col)
    data = data[[c for c in data if c not in col] + col]
    return data


# Searches a list of text arguments and returns a list that includes/excludes
# the list elements that contain the keywords
def keywordSearch(text_list, include_keywords, exclude_keywords=None):
    """
    Searches a list of text arguments and returns a list that includes/excludes
    the list elements that contain the keywords.
    
    Useful when searching for column names in a large dataframe, for example.

    Parameters
    ----------
    text_list : list
        List of string values.
    include_keywords : list
        List of keywords to search for.
    exclude_keywords : list, optional
        List of keywords that might be similar to include_keywords, but should
        be excluded nonetheless. The default is None.

    Returns
    -------
    found_list : list
        Returns items from the original list that match the keywords.

    """
    found_list = []
    for text in text_list:
        is_in_i_keys = sum((1 if i_key in text else 0 for i_key in include_keywords))
        is_in_e_keys = 0
        if exclude_keywords is not None:
            is_in_e_keys = sum((1 if e_key in text else 0 for e_key in exclude_keywords))
        if is_in_i_keys > 0 and is_in_e_keys == 0:
            found_list.append(text)
    return found_list


def toFrame(todf, fromdf):
    """
    Convert numpy array to pandas dataframe using past dataframe structure. 

    Useful when you want to store the dataframe values as a seperate numpy
    array, and make adjustments to the values, but want to convert back to a
    dataframe once the calculations are completed. 

    Parameters
    ----------
    todf : numpy array
        Data to convert to pandas dataframe.
    fromdf : pandas dataframe
        Dataframe containing structure to convert back to.

    Returns
    -------
    pandas dataframe
        The converted dataframe.

    """
    return pd.DataFrame(todf, index=fromdf.index, columns=fromdf.columns)


def inferFeatureType(X, n_unique=None):
    r"""Infer categorical, numeric, or date from a one-dimensional column's dtype.

    Parameters
    ----------
    X : pandas.Series or one-dimensional array-like
        Column whose feature type is required. Public pandas dtype checks also
        handle nullable extension dtypes.
    n_unique : int, optional
        Maximum distinct nonmissing count for treating a numeric column as
        categorical. A larger count remains numeric. If None, no numeric
        category cutoff is applied. The default is None.

    Returns
    -------
    d_type : {'categorical', 'numeric', 'date'}
        Inferred feature type. Booleans and nonnumeric columns are categorical.
        Datetime and timedelta columns use date.

    Notes
    -----
    Missing values are excluded from the unique-count cutoff. For example,
    [1, 1, NaN] has one distinct nonmissing value. This rule is independent of
    whether a caller separately reports missingness.

    For inspecting raw NumPy arrays, dtype.kind codes include b for boolean,
    i for signed integer, u for unsigned integer, f for floating point, c for
    complex, m for timedelta, M for datetime, O for object, S for bytes, U for
    Unicode, and V for void data. These kind codes differ from dtype.char:
    for example, the character code for boolean is ?, while b can describe a
    signed byte. Feature inference uses pandas dtype checks rather than these
    character-code shortcuts.
    """
    from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_timedelta64_dtype, is_numeric_dtype
    series = X if isinstance(X, pd.Series) else pd.Series(X)
    if is_datetime64_any_dtype(series.dtype) or is_timedelta64_dtype(series.dtype):
        return 'date'
    if is_bool_dtype(series.dtype) or not is_numeric_dtype(series.dtype):
        return 'categorical'
    if n_unique is not None and series.nunique(dropna=True) <= n_unique:
        return 'categorical'
    return 'numeric'


def reshape_to_vector(ar, axis=1):
    """
    Flatten or reshape an array to be a vector (or transpose an already
    flat array).

    Parameters
    ----------
    ar : numpy array
        An array to be reshaped.
    axis : int, optional
        The direction to flatten. The default is 1.

    Returns
    -------
    ar : numpy array
        The flattened array.
    """
    # TODO: Need to look at the use case for this again. There may be a more
    # efficient way to do this.
    if len(ar.shape) == 1:
        if axis == 1:
            return ar.reshape(ar.shape[0], 1)
        elif axis == 0:
            return ar.reshape(1, ar.shape[0])
        else:
            raise ValueError('Invalid axis dimension, either 0 or 1')
    return ar


def decision_boundary_1D(x, thres=0.5):
    """
    Divides probabilities `x` into two classes based on threshold value. Everything
    above the threshold is the positive class.
    """
    y = np.zeros(shape=(len(x), 1))
    y[x >= thres] = 1
    return y


def decision_boundary(x):
    """
    Divides probabilities `x` into classes based on the max probability value.
    """
    y = np.zeros(shape=x.shape)
    y[np.arange(len(x)), x.argmax(1)] = 1
    return y


def one_hot_encode(x):
    """
    The first column is the 0th class, 2nd column is the 1st class, etc...
    """
    n = np.max(x) + 1
    return np.eye(n)[x]


def one_hot_decode(x):
    """
    The first column is the 0th class, 2nd column is the 1st class, etc...
    """
    return np.argmax(x, axis=1)


def sample(array, nsamples, prob=None, replace=False, random_state=None):
    r"""Draw a weighted sample through an interface similar to R's sample.

    Parameters
    ----------
    array : list or list-like
        Population to sample from.
    nsamples : int
        Number of values to draw.
    prob : list or list-like, optional
        Finite, nonnegative relative weights, one per population value. Weights
        need not sum to one. If None, all values have equal weight.
    replace : bool, optional
        Draw with replacement when True. The default is False.
    random_state : int, optional
        Seed a local Python Random instance for repeatable results. None uses
        a fresh local generator. The default is None.

    Returns
    -------
    samp : list
        Sampled population values, in draw order.

    Raises
    ------
    ValueError
        If weight count or values are invalid, total weight is not positive,
        the requested sample size is negative, or too few positive-weight
        population values are available without replacement.

    Notes
    -----
    Caller weights are copied. Each draw uses cumulative remaining weights;
    without replacement, the selected value's weight is set to zero. Zero-weight
    values are never selected. Calls do not change the module-level random
    state, and random.seed does not control an unseeded call. Python and R use
    different random-number streams, so equal seeds do not imply equal samples.
    """
    weights = np.ones(len(array), dtype=float) if prob is None else np.array(prob, dtype=float, copy=True)
    if weights.shape != (len(array),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Supply one finite, nonnegative weight per value')
    if nsamples < 0 or weights.sum() <= 0:
        raise ValueError('The sample size must be nonnegative and total weight must be positive')
    if not replace and nsamples > np.count_nonzero(weights):
        raise ValueError('The sample size exceeds the population with positive weights')
    rng = random.Random(random_state)
    result = []
    for _ in range(nsamples):
        cumulative = np.cumsum(weights)
        index = bisect_right(cumulative, rng.random() * cumulative[-1])
        result.append(array[index])
        if not replace:
            weights[index] = 0
    return result
