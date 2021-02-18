from __future__ import (division, generators, absolute_import,
                        print_function, with_statement, nested_scopes,
                        unicode_literals)
import datetime
import numpy as np
import os
import pandas as pd
import pickle
import sys

from getpass import getpass
from sqlalchemy import create_engine
from timeit import default_timer as dt


def dsn_getTable(sql, user, dialect, dsn, pw=None, create_engine_args={},
                 read_sql_args={}):
    '''
    Load data into python from a sql database. 
    
    Requires a DSN or connection string. Uses sqlAlchemy to make the
    connection, and closes the connection on completion.
    See https://docs.sqlalchemy.org/en/13/core/engines.html for
    help building connection strings.

    Parameters
    ----------
    sql : str, or list
        A string containing the sql statement to be ran (It should be something
        that returns a table), or a list of sql statements to return multiple
        tables (tables are returned as a list of pandas dataframes, in this
        case).
    user : str
        The username for the database application, if applicable.
    dialect : str
        The database flavor (oracle, mysql, etc.)
    dsn : str, optional
        Can either be a saved DSN from your computer, or a string that
        represents all the relevant information provided in a DSN. 
        It can also be a host:port combination. See sqlalchemy
        documentation for more details.
    pw : str, optional
        The password for the database application that matches the user
        specified, if applicable. If None or '', getpass will ask for a
        password. If there is no password, just leave it blank.
        The default is None.
    create_engine_args : dict
        A dictionary containing arguments to be passed to sqlAlchemy's
        create_engine function.
    read_sql_args : dict
        A dictionary containing arguments to be passed to pandas'
        read_sql function.

    Returns
    -------
    data : pandas dataframe or list
        A dataframe (or list of dataframes) containing the returned data.

    '''
    if pw is None or pw == '':
        pw = getpass(prompt='Password: ')
    
    # TODO: include better support for sqlite, since it has a weird
    # OS-dependent connection string.
    conn = '{dialect}://{user}:{password}@{dsn}'.format(
        dialect=dialect,
        user=user,
        password=pw,
        dsn=dsn
    )

    engine = create_engine(conn, **create_engine_args)

    data = None
    if isinstance(sql, str):
        data = pd.read_sql(sql=sql, con=engine, **read_sql_args)
    else:
        data = []
        for query in sql:
            data.append(pd.read_sql(sql=query, con=engine, **read_sql_args))
    engine.dispose()
    
    return data


def dsn_saveTable(data, tableName, user, dialect, dsn, pw=None,
                  create_engine_args={}, to_sql_args={}):
    '''
    Save pandas dataframe to database.

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
    dsn : str, optional
        Can either be a saved DSN from your computer, or a string that
        represents all the relevant information provided in a DSN. 
        It can also be a host:port combination. See sqlalchemy
        documentation for more details.
    pw : str, optional
        The password for the database application that matches the user
        specified, if applicable. If None or '', getpass will ask for a
        password. If there is no password, just leave it blank.
        The default is None.
    create_engine_args : dict
        A dictionary containing arguments to be passed to sqlAlchemy's
        create_engine function. Default is {}
    to_sql_args : dict
        A dictionary containing arguments to be passed to pandas'
        read_sql function. Default is {}

    Returns
    -------
    None.

    '''    
    if pw is None or pw == '':
        pw = getpass(prompt='Password: ')
    
    # TODO: include better support for sqlite, since it has a weird
    # OS-dependent connection string.
    conn = '{dialect}://{user}:{password}@{dsn}'.format(
        dialect=dialect,
        user=user,
        password=pw,
        dsn=dsn
    )

    engine = create_engine(conn, **create_engine_args)  
    
    data.to_sql(name=tableName, con=engine, **to_sql_args)
    engine.dispose()
    
    return None


def pretty_print_time(ts, te=None, decimals=0):
    '''
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

    '''
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
    '''
    Check if a directory exists. If it doesn't, create it.

    Parameters
    ----------
    dir : str
        Directory to check.
    make : bool, optional
        Whether or not to create a missing directory. The default is True.
    verbose : bool
        Print results.

    Returns
    -------
    bool
        True if the directory exists.

    '''
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
    '''
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

    '''
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
    checkDir(log_dir)
    logfile = log_dir + "/" + log_filename
    logfile = logfile.replace("//", "/")
    if log:
        with open(logfile, log_args) as f:
            f.write(printed)
    return printed
    

def print_time(*args, **kwargs):
    '''
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

    '''
    print_message_with_time(*args, **kwargs)


def save_checkpoint(obj, msg, ts=None, te=None, decimals=0,
                    display_realtime=True, backsn=False,
                    log=True, log_dir="log", log_filename="", log_args="a",
                    time_first=False, checkpoint_dir="checkpoints",
                    checkpoint_filename="chkpnt", checkpoint_extension="pkl",
                    checkpoint_args="wb"):
    '''
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

    '''
    print_message_with_time(msg=msg, ts=ts, te=te, decimals=decimals,
                            display_realtime=display_realtime,
                            backsn=backsn, log=log, log_dir=log_dir,
                            log_filename=log_filename,
                            log_args=log_args)
    checkDir(checkpoint_dir)
    file = checkpoint_dir + '/' + checkpoint_filename + "." + checkpoint_extension
    file = file.replace('//', '/')
    if checkpoint_extension == 'pkl':
        with open(file, checkpoint_args) as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def updateProgBar(curIter, totalIter, t0, barLength=20, decimals=0):
    '''
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

    '''
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
    '''
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

    '''
    if not isinstance(col, list):
        col = list(col)
    data = data[[c for c in data if c not in col] + col]
    return data


# Searches a list of text arguments and returns a list that includes/excludes
# the list elements that contain the keywords
def keywordSearch(text_list, include_keywords, exclude_keywords=None):
    '''
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

    '''
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
    '''
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

    '''
    return pd.DataFrame(todf, index=fromdf.index, columns=fromdf.columns)


def inferFeatureType(X, n_unique=None):
    '''
    Infer feature type of a column based on it's `dytpe`. The inferred
    datatypes are `categorical`, `numeric`, or `date`.

    
    Parameters
    ----------
    X : dataframe or ndarray
        The dataframe containing columns whose datatype needs to be inferred.
    n_unique : int
        The number of unique values to use as a cutoff for numeric columns,
        for example, if the number of unique values for a numeric column is
        greater than n_unique, consider it numeric. Default is None.
    
    Returns:
    --------
    d_type : str
        String representation of the d_type
    
    numpy arrays.dtypes reference table:
    
    ? : boolean
    b : signed byte (sometimes interchangeable with boolean, it seems)
    B : unsigned byte
    i : signed integer
    u : unsigned integer
    f : float
    c : complex float
    m : timedelta
    M : datetime
    O : object
    S : zero-terminated bytes
    a : zero-terminated bytes
    U : unicode string
    V : raw data (void)
    
    '''
    d_type = None
    if X.dtype.kind in "OSUVaBb": # ?/b numeric or categorical?
        d_type = "categorical"
    elif X.dtype.kind in "mM":
        d_type = "date"
    else:
        if n_unique is None:
            d_type = "numeric"
        else:
            n_uni_obs = len(pd.unique(X))
            if n_uni_obs <= n_unique:
                d_type = "categorical"
            else:
                d_type = "numeric"
    return d_type


def reshape_to_vect(ar, axis=1):
    '''
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

    '''
    # TODO: Need to look at the use case for this again. There may be a more
    # efficient way to do this.
    if len(ar.shape) == 1:
        if axis == 1:
            return ar.reshape(ar.shape[0], 1)
        elif axis == 0:
            return ar.reshape(1, ar.shape[0])
        else:
            raise('Invalid axis dimension, either 0 or 1')
    return ar


def decision_boundary_1D(x, thres=0.5):
    '''
    Divides probabilities `x` into two classes based on threshold value. Everything
    above the threshold is the positive class.
    '''
    y = np.zeros(shape=(len(x), 1))
    y[x >= thres] = 1
    return y


def decision_boundary(x):
    '''
    Divides probabilities `x` into classes based on the max probability value.
    '''
    y = np.zeros(shape=x.shape)
    y[np.arange(len(x)), x.argmax(1)] = 1
    return y


def one_hot_encode(x):
    '''
    The first column is the 0th class, 2nd column is the 1st class, etc...
    '''
    n = np.max(x) + 1
    return np.eye(n)[x]


def one_hot_decode(x):
    '''
    The first column is the 0th class, 2nd column is the 1st class, etc...
    '''
    return np.argmax(x, axis=1)
