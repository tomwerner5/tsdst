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


def pde_getTable(sql, user, pw=None, convert_unicode=False, pool_recyle=10,
                 pool_size=20, echo=False, dsn=None):
                 #max_identifier_length=128):
    if dsn is None:
	# The default is oracle, since that's what I was using at the time
        dsn = """
                (DESCRIPTION =
                    (ADDRESS =
                        (PROTOCOL = TCP)
                    (HOST = 10.10.10.10)
                    (PORT = 1234)
                )
                (CONNECT_DATA =
                    (SERVICE_NAME = someschema)
                ))
            """
    
    if pw is None or pw == '':
        pw = getpass(prompt='Password: ')
    
    conn = 'oracle://{user}:{password}@{dsn}'.format(
        user=user,
        password=pw,
        dsn=dsn
    )

    engine = create_engine(conn, convert_unicode=convert_unicode,
                           pool_recycle=pool_recyle,
                           pool_size=pool_size, echo=echo)
                           #,max_identifier_length=max_identifier_length)
    data = None
    if isinstance(sql, str):
        data = pd.read_sql(sql, engine)
    else:
        data = []
        for query in sql:
            data.append(pd.read_sql(query, engine))
    engine.dispose()
    
    return data


def pde_saveTable(data, tableName, user, pw=None, schema=None, dsn=None,
                  if_exists='fail', write_index_as_col=True,
                  index_label=None, chunksize=None, dtype=None,
                  method=None, convert_unicode=False, pool_recyle=10,
                  pool_size=20, echo=False):
                  #max_identifier_length=128):
    if dsn is None:
	# The default is oracle, since that's what I was using at the time
        dsn = """
                (DESCRIPTION =
                    (ADDRESS =
                        (PROTOCOL = TCP)
                    (HOST = 10.10.10.10)
                    (PORT = 1234)
                )
                (CONNECT_DATA =
                    (SERVICE_NAME = someschema)
                ))
            """
    
    if pw is None or pw == '':
        pw = getpass(prompt='Password: ')
    
    conn = 'oracle://{user}:{password}@{dsn}'.format(
        user=user,
        password=pw,
        dsn=dsn
    )

    engine = create_engine(conn, convert_unicode=convert_unicode,
                           pool_recycle=pool_recyle,
                           pool_size=pool_size, echo=echo)
                           #,max_identifier_length=max_identifier_length)
  
    
    pd.to_sql(name=name, con=engine, schema=schema, if_exists=if_exists,
              index=write_index_as_col, index_label=index_label,
              cunksize=chunksize, dtype=dtype, method=method)
    engine.dispose()
    
    return None


def pretty_print_time(ts, te=None):
    if te is None:
        t = ts
    else:
        t = te - ts
    h = np.floor(t/3600.0)
    m = np.floor(((t/3600.0) - h) * 60.0)
    s = ((((t/3600.0) - h) * 60.0) - m) * 60.0
    
    if len(str(int(h))) == 1:
        h = "0" + str(int(h))
    else:
        h = str(int(h))
    
    if len(str(int(m))) == 1:
        m = "0" + str(int(m))
    else:
        m = str(int(m))
    
    if len(str(int(np.round(s)))) == 1:
        s = "0" + str(int(np.round(s, 3)))
    else:
        s = str(int(np.round(s, 3)))
    
    pretty = h + ":" + m + ":" + s
    return pretty


def checkDir(dirc, make=True):
    if not os.path.isdir(dirc):
        if make:
            os.mkdir(dirc)  
        return False
    else:
        return True

def print_message_with_time(msg, ts, te=None, display_realtime=True, backsn=False,
                            log=False, log_dir="log", log_filename="", log_args="a",
                            time_first=False):
    
    date_time = datetime.datetime.now().strftime("%-I:%M:%S %p (%b %d)")
    if display_realtime:
        time_str = pretty_print_time(ts, te) + " Current Time: " + date_time
    else:
        time_str = pretty_print_time(ts, te)
    if time_first:
        printed = time_str + msg
    else:
        printed = msg + time_str
    if backsn:
        printed = printed + "\n"
    sys.stdout.write(printed)
    sys.stdout.flush()
    checkDir(log_dir)
    logfile = log_dir + "/" + log_filename + ".log"
    logfile = logfile.replace("//", "/")
    if log:
        with open(log_dir + "/" + log_filename + ".log", log_args) as f:
            f.write(printed)
    return printed
    

def print_time(*args, **kwargs):
    print_message_with_time(*args, **kwargs)


def save_checkpoint(obj, msg, ts=None, te=None, display_realtime=True, backsn=False,
                    log=True, log_dir="log", log_filename="", log_args="a",
                    checkpoint_dir="checkpoints", checkpoint_filename="",
                    checkpoint_extension="pkl", checkpoint_args="wb"):
    print_message_with_time(msg=msg, ts=ts, te=te, display_realtime=display_realtime,
                            backsn=backsn, log=log, log_dir=log_dir, log_filename=log_filename,
                            log_args=log_args)
    checkDir(checkpoint_dir)
    file = checkpoint_dir + '/' + checkpoint_filename + "." + checkpoint_extension
    file = file.replace('//', '/')
    if checkpoint_extension == 'pkl':
        with open(file, checkpoint_args) as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def updateProgBar(curIter, totalIter, t0, barLength=20):
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
        round(progress*100.0, 2), curIter, totalIter, status, pretty_print_time(t0, dt()),
        pretty_print_time((dt()-t0)/curIter * (totalIter - curIter)))
    if progress >= 1:
        sys.stdout.write(text + "\r\n")
        sys.stdout.flush()
    else:
        sys.stdout.write(text)
        sys.stdout.flush()
        
        
def move_columns_to_end(data, col):
    if not isinstance(col, list):
        col = list(col)
    data = data[[c for c in data if c not in col] + col]
    return data


# Searches a list of text arguments and returns a list that includes/excludes
# the list elements that contain the keywords
def keywordSearch(text_list, include_keywords, exclude_keywords=None):
    found_list = []
    for text in text_list:
        present = False
        is_in_i_keys = sum((1 if i_key in text else 0 for i_key in include_keywords))
        is_in_e_keys = 0
        if exclude_keywords is not None:
            is_in_e_keys = sum((1 if e_key in text else 0 for e_key in exclude_keywords))
        if is_in_i_keys > 0 and is_in_e_keys == 0:
            found_list.append(text)
    return found_list


def splitCol(data, search_col, search_list, split_pat):
    exists = []
    noneCount = 0
    for i in data.index:
        if data.loc[i, search_col] is None:
            noneCount += 1
        else:
            found_splits = data.loc[i, search_col].split(split_pat)
            exists.append([1 if val in found_splits else 0 for val in search_list])
    return exists, noneCount


def toFrame(todf, fromdf):
    return pd.DataFrame(todf, index=fromdf.index, columns=fromdf.columns)


def inferFeatureType(X, n_unique=None):
    '''
    Not all of the Datatypes are stored in the metadata tables of Bedrock, so some need to be inferred.
    These inferred/derived datatypes are used to simply the plotting mechanisms
    
    X: dataframe or ndarray
    n_unique: int
    
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
                dtype = "categorical"
            else:
                d_type = "numeric"
    return d_type
