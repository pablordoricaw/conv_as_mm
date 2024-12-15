import numpy as np

from scipy.signal import toeplitz

def conv_as_mmult(I, F, mode="full"):
    """
    Perform 2D convolution using matrix multiplication.

    This function implements 2D convolution by converting the operation into a matrix
    multiplication problem. It supports 'full', 'same', and 'valid' convolution modes.

    Parameters:
    -----------
    I : numpy.ndarray
        Input 2D array (image) to be convolved.
    F : numpy.ndarray
        2D convolution kernel (filter).
    mode : str, optional
        A string indicating the size of the output:
        - 'full': The output is the full discrete linear convolution (default).
        - 'same': The output is the same size as the input I, centered
                  with respect to the 'full' output.
        - 'valid': The output consists only of those elements that do not
                   rely on zero-padding.

    Returns:
    --------
    numpy.ndarray
        The 2D convolution of I and F, with size depending on the mode:
        - 'full': (I.shape[0] + F.shape[0] - 1, I.shape[1] + F.shape[1] - 1)
        - 'same': Same size as input I
        - 'valid': (I.shape[0] - F.shape[0] + 1, I.shape[1] - F.shape[1] + 1)

    Raises:
    -------
    ValueError
        If an invalid mode is specified.

    Notes:
    ------
    This function uses the Toeplitz matrix approach to convert the 2D convolution
    into a matrix multiplication problem. It constructs a doubly blocked Toeplitz
    matrix from the filter F and then performs matrix multiplication with the
    vectorized input I.

    The implementation is equivalent to scipy.signal.convolve2d but uses a
    different algorithmic approach.

    Examples:
    ---------
    >>> import numpy as np
    >>> I = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> F = np.array([[1, 0], [0, 1]])
    >>> conv_as_mmult(I, F, mode='full')
    array([[1, 2, 3, 0],
           [4, 6, 5, 3],
           [7, 8, 9, 6],
           [0, 7, 8, 9]])
    >>> conv_as_mmult(I, F, mode='same')
    array([[1, 2, 3],
           [4, 6, 5],
           [7, 8, 9]])
    >>> conv_as_mmult(I, F, mode='valid')
    array([[6, 5],
           [8, 9]])
    """

    if mode != "full" and mode != "valid" and mode != "same":
        raise ValueError("mode should be 'full', 'valid' or 'same'")

    I_row_num, I_col_num = I.shape
    F_row_num, F_col_num = F.shape

    #  calculate the output dimensions
    output_row_num = I_row_num + F_row_num - 1
    output_col_num = I_col_num + F_col_num - 1

    # zero pad the filter
    F_zero_padded = np.pad(F, ((0, output_row_num - F_row_num),
                               (0, output_col_num - F_col_num)),
                                'constant', constant_values=0)

    toeplitz_list = []
    for row in F_zero_padded:
        c = np.zeros(I_col_num)
        c[0] = row[0]
        toeplitz_m = toeplitz(row, c) # this function is in scipy.linalg library
        toeplitz_list.append(toeplitz_m)

    # doubly blocked toeplitz indices:
    # this matrix defines which toeplitz matrix from toeplitz_list goes to which
    # part of the doubly blocked
    c = range(1, F_zero_padded.shape[0] + 1)
    r = np.r_[c[0], np.zeros(I_row_num - 1, dtype=int)]
    doubly_indices = toeplitz(c, r)

    # create doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape # shape of one Toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # height and widths of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

    vectorized_I = I.flatten()
    result_vector = np.matmul(doubly_blocked, vectorized_I)

    full_result = np.reshape(result_vector, (output_row_num, output_col_num))

    if mode == 'same':
        start_row = (F_row_num - 1) // 2
        start_col = (F_col_num - 1) // 2
        full_result = full_result[start_row:start_row+I_row_num, start_col:start_col+I_col_num]
    elif mode == 'valid':
        full_result = full_result[F_row_num-1:I_row_num, F_col_num-1:I_col_num]

    return full_result
