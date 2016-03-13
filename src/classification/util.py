import numpy as np
from scipy.sparse import csr_matrix
import json


def read_file_linewise_2_array(file):
    f = open(file)
    a = []
    for l in f:
        a.append(l.strip())
    return a

def group_and_count(l, key=None):
    if key is not None:
        l = map(key, l)
    d = {}
    for item in l:
        if item not in d:
            d[item] = 0
        d[item] = d[item] + 1
    return d

# csr matrixes
def save_csr_matrix(mat, filename):
    """Saves the given csr_matrix under the given filename. The filename will probably end with '.npz'"""
    np.savez(filename, data=mat.data, indices=mat.indices,
             indptr=mat.indptr, shape=mat.shape)


def load_csr_matrix(filename):
    """Loads and returns the csr_matrix at the given filename."""
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def build_csr_matrix(list_of_dicts, token2index_map=None, num_attributes=None):
    """Returns a csr_matrix out of a list of dicts where each dict will correspond to a row in the created matrix.
    Either token2index_map or num_attributes has to be set (but not both). The token2indey_map maps a token
    (e.g. a text token) to a column index. If it is set each dict in the list of dicts must map tokens to certain a
    value. e.g.
        token2index_map = {'foo': 0, 'bar': 1, 'baz': 2}
        list_of_dicts = [{'foo': 0.3, 'hello': -1.1}, {'bar': 0.5, 'baz': 0.5}, {'bar': 2.2, 'hello': 0.6, 'test': -0.2}]
    will produce a matrix like:
        [ [0.3 0.0 0.0],
          [0.0 0.5 0.5],
          [0.0 2.2 0.0]]

    num_attribtes deternime the number of colums in the resulting matrix. If num_attributes is set, the dicts in
    the list_of_dicts must map directly from a column index to a value. e.g.

        num_attributes = 4
        list_of_dicts = [{0: 0.3, 2: 1.1}, {1: 0.5, 2: 0.5}, {2: 2.2}]
    will produce a matrix like:
        [ [0.3 0.0 1.1 0.0],
          [0.0 0.5 0.5 0.0],
          [0.0 0.0 2.2 0.0]]"""

    if token2index_map is None and num_attributes is None:
        raise ValueError("either token2index_map or num_attributes must be set")
    elif (token2index_map is not None) and (num_attributes is not None):
        raise ValueError("token2index_map and num_attributes cannot both be set")
    else:
        pass

    row = []
    col = []
    data = []

    i = 0
    for m in list_of_dicts:
        numerical_sorted_tokens = None

        if token2index_map is not None:
            tokens_in_dict = filter(lambda kv: kv[0] in token2index_map, m.items())
            translated_tokens = map(lambda kv: (token2index_map[kv[0]], kv[1]), tokens_in_dict)
            numerical_sorted_tokens = sorted(translated_tokens, key=lambda x: x[0])
        elif num_attributes is not None:
            numerical_sorted_tokens = sorted(m.items(), key=lambda x: x[0])
        else:
            raise ValueError("Invalid case while building csr_matrix")

        for key, val in numerical_sorted_tokens:
            row.append(i)
            col.append(key)
            data.append(val)

        i += 1

    shape_rows = len(list_of_dicts)
    shape_cols = None
    if token2index_map is not None:
        shape_cols = len(token2index_map)
    elif num_attributes is not None:
        shape_cols = num_attributes
    else:
        raise ValueError("Invalid case while building csr_matrix")

    return csr_matrix((data, (row, col)), shape=(shape_rows, shape_cols))


def parent_class(msc_class):
    if len(msc_class) == 2:
        return "root"
    elif len(msc_class) == 3:
        return msc_class[:2]
    elif len(msc_class) == 5:
        return msc_class[:3]
    else:
        raise ValueError("Invalid msc class: " + str(msc_class))