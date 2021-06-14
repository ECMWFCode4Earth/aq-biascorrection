import numpy as np
import xarray as xr
import xskillscore as xs
from scipy import special


def validate_weights(da, dim, weights):
    if not isinstance(weights, xr.DataArray):
        raise TypeError(
            f'You provided weights with type={type(weights)}.\n'
            'Weights must be an xarray.DataArray with shape that is broadcastable \n'
            'to shape= {da.data.shape} of da = {da}.'
        )
    # if NaN are present, we need to use individual weights
    total_weights = weights.where(da.notnull()).sum(dim=dim)

    # Make sure weights add up to 1.0
    rtol = 1e-6 if weights.dtype == np.float32 else 1e-7
    np.testing.assert_allclose((weights / weights.sum(dim)).sum(dim), 1.0, rtol=rtol)

    return weights, total_weights


def weighted_mean_da(da, dim=None, weights=None):
    """ Compute weighted mean for DataArray
    Parameters
    ----------
    da : xarray.DataArray
        DataArray for which to compute `weighted mean`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply mean.
    weights : xarray.DataArray or array-like
        weights to apply. Shape must be broadcastable to shape of da.
    Returns
    -------
    reduced : xarray.DataArray
        New DataArray with mean applied to its data and the indicated
        dimension(s) removed.
    """
    if dim is None:
        dim = list(da.dims)
    if weights is None:
        return da.mean(dim)

    elif all(d in da.dims for d in dim):
        weights, total_weights = validate_weights(da, dim, weights)
        return (da * weights).sum(dim) / total_weights

    else:
        return da


def weighted_mean(data, dim=None, weights=None):
    """ Compute weighted mean for xarray objects
    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
         xarray object for which to compute `weighted mean`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply mean.
    weights : xarray.DataArray or array-like
        weights to apply. Shape must be broadcastable to shape of data.
    Returns
    -------
    reduced : xarray.Dataset or xarray.DataArray
        New xarray object with weighted mean applied to its data and the indicated
        dimension(s) removed.
    """

    if isinstance(dim, str):
        dim = [dim]

    if isinstance(data, xr.DataArray):
        return weighted_mean_da(data, dim, weights)
    elif isinstance(data, xr.Dataset):
        return data.apply(weighted_mean_da, dim=dim, weights=weights)

    else:
        raise ValueError('Data must be an xarray Dataset or DataArray')


def weighted_rmsd(x, y, dim=None, weights=None):
    """ Compute weighted root mean square deviation between two xarray DataArrays
    Parameters
    ----------
    x, y : xarray.DataArray
        xarray DataArray for which to compute `weighted_rmsd`.
    dim : str or sequence of str, optional
        Dimension(s) over which to apply rmsd.
    weights : xarray.DataArray or array-like
        weights to apply. Shape must be broadcastable to shape of data.
    Returns
    -------
    reduced : xarray.DataArray
        New DataArray with root mean square deviation applied to x, y and the indicated
        dimension(s) removed.
    """

    if isinstance(dim, str):
        dim = [dim]

    if not isinstance(x, xr.DataArray) or not isinstance(y, xr.DataArray):
        raise ValueError('x and y must be xarray DataArrays')
    dev = (x - y) ** 2
    dev_mean = weighted_mean(dev, dim, weights)
    return np.sqrt(dev_mean)


def rmse(x, y, dim):
    """
    Compute Root Mean Squared Error.
    Parameters
    ----------
    x : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    y : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    dim : str
        The dimension to apply the correlation along.
    Returns
    -------
    Root Mean Squared Error
        Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
        numpy.ndarray, the first type on that list to appear on an input.
    """

    return xs.rmse(x, y, dim)


def mse(x, y, dim):
    """
    Compute Mean Squared Error.
    Parameters
    ----------
    x : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    y : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    dim : str
        The dimension to apply the correlation along.
    Returns
    -------
    Mean Squared Error
        Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
        numpy.ndarray, the first type on that list to appear on an input.
    """

    return xs.mse(x, y, dim)


def pearson(x, y, dim):
    """
        Compute Pearson Correlation
        Parameters
        ----------
        x : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
            Mix of labeled and/or unlabeled arrays to which to apply the function.
        y : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
            Mix of labeled and/or unlabeled arrays to which to apply the function.
        dim : str
            The dimension to apply the correlation along.
        Returns
        -------
        Pearson Correlation
            Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
            numpy.ndarray, the first type on that list to appear on an input.
        """
    return xs.pearson_r(x, y, dim)


def weighted_corr(x, y, dim=None, weights=None, return_p=True):
    """ Compute weighted correlation between two `xarray.DataArray`.
    Parameters
    ----------
    x, y : xarray.DataArray
        xarray DataArrays for which to compute `weighted correlation`.
    dim : str or sequence of str, optional
        Dimension(s) over which to apply correlation.
    weights : xarray.DataArray or array-like
        weights to apply. Shape must be broadcastable to shape of data.
    return_p : bool, default: True
        If True, compute and return the p-value(s) associated with the
        correlation.
    Returns
    -------
    reduced : xarray.DataArray
        New DataArray with correlation applied to x, y and the indicated
        dimension(s) removed.
        If `return_p` is True, appends the resulting p values to the
        returned Dataset.
    """

    if isinstance(dim, str):
        dim = [dim]

    if not isinstance(x, xr.DataArray) or not isinstance(y, xr.DataArray):
        raise ValueError('x and y must be xarray DataArrays')

    valid_values = x.notnull() & y.notnull()
    x = x.where(valid_values)
    y = y.where(valid_values)
    numerator = weighted_cov(x, y, dim, weights)
    denominator = np.sqrt(weighted_cov(x, x, dim, weights) * weighted_cov(y, y, dim, weights))
    corr_xy = numerator / denominator

    if return_p:
        p = compute_corr_significance(corr_xy, len(x))
        corr_xy.name = 'r'
        p.name = 'p'
        return xr.merge([corr_xy, p])
    else:
        return corr_xy


def compute_corr_significance(r, N):
    """ Compute statistical significance for a pearson correlation between
        two xarray objects.
    Parameters
    ----------
    r : `xarray.DataArray` object
        correlation coefficient between two time series.
    N : int
        length of time series being correlated.
    Returns
    -------
    pval : float
        p value for pearson correlation.
    """
    df = N - 2
    t_squared = r ** 2 * (df / ((1.0 - r) * (1.0 + r)))
    # method used in scipy, where `np.fmin` constrains values to be
    # below 1 due to errors in floating point arithmetic.
    pval = special.betainc(0.5 * df, 0.5, np.fmin(df / (df + t_squared), 1.0))
    return xr.DataArray(pval, coords=t_squared.coords, dims=t_squared.dims)


def weighted_cov(x, y, dim=None, weights=None):
    """ Compute weighted covariance between two xarray DataArrays.
    Parameters
    ----------
    x, y : xarray.DataArray
        xarray DataArrays for which to compute `weighted covariance`.
    dim : str or sequence of str, optional
        Dimension(s) over which to apply covariance.
    weights : xarray.DataArray or array-like
        weights to apply. Shape must be broadcastable to shape of data.
    Returns
    -------
    reduced : DataArray
        New DataArray with covariance applied to x, y and the indicated
        dimension(s) removed.
    """

    if isinstance(dim, str):
        dim = [dim]

    if not isinstance(x, xr.DataArray) or not isinstance(y, xr.DataArray):
        raise ValueError('x and y must be xarray DataArrays')
    mean_x = weighted_mean(x, dim, weights)
    mean_y = weighted_mean(y, dim, weights)
    dev_x = x - mean_x
    dev_y = y - mean_y
    dev_xy = dev_x * dev_y
    cov_xy = weighted_mean(dev_xy, dim, weights)
    return cov_xy
