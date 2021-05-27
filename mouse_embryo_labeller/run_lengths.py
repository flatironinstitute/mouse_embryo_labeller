
import numpy as np

def run_length_encode(array, chunksize=((1 << 16) - 1), dtype=np.int16):
    "Chunked run length encoding for very large arrays containing smallish values."
    shape = array.shape
    ravelled = array.ravel()
    length = len(ravelled)
    chunk_cursor = 0
    runlength_chunks = []
    while chunk_cursor < length:
        chunk_end = chunk_cursor + chunksize
        chunk = ravelled[chunk_cursor : chunk_end]
        chunk_length = len(chunk)
        change = (chunk[:-1] != chunk[1:])
        change_indices = np.nonzero(change)[0]
        nchanges = len(change_indices)
        cursor = 0
        runlengths = np.zeros((nchanges + 1, 2), dtype=dtype)
        for (count, index) in enumerate(change_indices):
            next_cursor = index + 1
            runlengths[count, 0] = chunk[cursor] # value
            runlengths[count, 1] = next_cursor - cursor # run length
            cursor = next_cursor
        # last run
        runlengths[nchanges, 0] = chunk[cursor]
        runlengths[nchanges, 1] = chunk_length - cursor
        runlength_chunks.append(runlengths)
        chunk_cursor = chunk_end
    all_runlengths = np.vstack(runlength_chunks)
    description = dict(
        shape=shape,
        runlengths=all_runlengths,
        dtype=dtype,
        )
    return description

def run_length_decode(description):
    dtype = description["dtype"]
    runlengths = description["runlengths"]
    shape = description["shape"]
    array = np.zeros(shape, dtype=dtype)
    ravelled = array.ravel()
    cursor = 0
    for (value, size) in runlengths:
        run_end = cursor + size
        ravelled[cursor : run_end] = value
        cursor = run_end
    array = ravelled.reshape(shape)  # redundant?
    return array

def testing():
    A = np.zeros((50,), dtype=np.uint16)
    A[20:30] = 10
    A[30:35] = 6
    A[40:] = 3
    test = run_length_encode(A, chunksize=17)
    B = run_length_decode(test)
    assert np.alltrue(A == B)
    print ("ok!")

if __name__=="__main__":
    testing()

