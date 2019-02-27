"""
Terrible functions I want to get rid off.
"""


def get_dimensions_encoding(n_bins, batchsize):
    """
    Returns a dimensions tuple for 2,3 and 4 dimensional data.
    :param int batchsize: Batchsize that is used in hdf5_batch_generator().
    :param tuple n_bins: Declares the number of bins for each dimension (x,y,z).
                        If a dimension is equal to 1, it means that the dimension should be left out.
    :return: tuple dimensions: 2D, 3D or 4D dimensions tuple (integers).
    """
    n_bins_x, n_bins_y, n_bins_z, n_bins_t = n_bins[0], n_bins[1], n_bins[2], n_bins[3]
    if n_bins_x == 1:
        if n_bins_y == 1:
            print('Using 2D projected data without dimensions x and y')
            dimensions = (batchsize, n_bins_z, n_bins_t, 1)
        elif n_bins_z == 1:
            print('Using 2D projected data without dimensions x and z')
            dimensions = (batchsize, n_bins_y, n_bins_t, 1)
        elif n_bins_t == 1:
            print('Using 2D projected data without dimensions x and t')
            dimensions = (batchsize, n_bins_y, n_bins_z, 1)
        else:
            print('Using 3D projected data without dimension x')
            dimensions = (batchsize, n_bins_y, n_bins_z, n_bins_t, 1)

    elif n_bins_y == 1:
        if n_bins_z == 1:
            print('Using 2D projected data without dimensions y and z')
            dimensions = (batchsize, n_bins_x, n_bins_t, 1)
        elif n_bins_t == 1:
            print('Using 2D projected data without dimensions y and t')
            dimensions = (batchsize, n_bins_x, n_bins_z, 1)
        else:
            print('Using 3D projected data without dimension y')
            dimensions = (batchsize, n_bins_x, n_bins_z, n_bins_t, 1)

    elif n_bins_z == 1:
        if n_bins_t == 1:
            print('Using 2D projected data without dimensions z and t')
            dimensions = (batchsize, n_bins_x, n_bins_y, 1)
        else:
            print('Using 3D projected data without dimension z')
            dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_t, 1)

    elif n_bins_t == 1:
        print('Using 3D projected data without dimension t')
        dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_z, 1)

    else:
        # print 'Using full 4D data'
        dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_z, n_bins_t)

    return dimensions
