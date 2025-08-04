use crate::ops::scan_parallel::{cumsum_dim_parallel, cumprod_dim_parallel};
use crate::{element::NdArrayElement, tensor::NdArrayTensor};
use burn_tensor::ElementConversion;
use core::cmp::PartialOrd;
use ndarray::Axis;

macro_rules! keepdim {
    (
        $dim:expr,
        $self:expr,
        mean
    ) => {{
        let tensor: NdArrayTensor<E> = mean_dim($self.clone(), $dim);
        let mut shape = $self.shape();
        shape.dims[$dim] = 1;
        NdArrayOps::reshape(tensor.clone(), shape)
    }};
    (
        $dim:expr,
        $self:expr,
        sum
    ) => {{
        let tensor: NdArrayTensor<E> = sum_dim($self.clone(), $dim);
        let mut shape = $self.shape();
        shape.dims[$dim] = 1;
        NdArrayOps::reshape(tensor, shape)
    }};
    (
        $dim:expr,
        $self:expr,
        prod
    ) => {{
        let tensor: NdArrayTensor<E> = prod_dim($self.clone(), $dim);
        let mut shape = $self.shape();
        shape.dims[$dim] = 1;
        NdArrayOps::reshape(tensor, shape)
    }};
}

pub(crate) use keepdim;

pub(crate) fn mean_dim<E: NdArrayElement>(
    tensor: NdArrayTensor<E>,
    dim: usize,
) -> NdArrayTensor<E> {
    let array = tensor.array.mean_axis(Axis(dim)).unwrap().into_shared();

    NdArrayTensor { array }
}

pub(crate) fn sum_dim<E: NdArrayElement>(tensor: NdArrayTensor<E>, dim: usize) -> NdArrayTensor<E> {
    let array = tensor.array.sum_axis(Axis(dim)).into_shared();

    NdArrayTensor { array }
}

pub(crate) fn prod_dim<E: NdArrayElement>(
    tensor: NdArrayTensor<E>,
    dim: usize,
) -> NdArrayTensor<E> {
    let array = tensor
        .array
        .fold_axis(Axis(dim), 1.elem::<E>(), |acc, &x| acc.mul(x.elem()))
        .into_shared();

    NdArrayTensor { array }
}

/// Cumulative sum along axis using parallel implementation when beneficial
pub(crate) fn cumsum_dim<E: NdArrayElement>(
    tensor: NdArrayTensor<E>,
    dim: usize,
) -> NdArrayTensor<E> {
    #[cfg(feature = "std")]
    {
        // Use parallel implementation for better multi-core utilization
        cumsum_dim_parallel(tensor, dim)
    }
    
    #[cfg(not(feature = "std"))]
    {
        // Fallback to sequential implementation
        let axis = Axis(dim);
        let mut array = tensor.array.into_owned();
        array.accumulate_axis_inplace(axis, |&prev, curr| *curr = *curr + prev);
        NdArrayTensor::new(array.into_shared())
    }
}

/// Cumulative product along axis using parallel implementation when beneficial
pub(crate) fn cumprod_dim<E: NdArrayElement>(
    tensor: NdArrayTensor<E>,
    dim: usize,
) -> NdArrayTensor<E> {
    #[cfg(feature = "std")]
    {
        // Use parallel implementation for better multi-core utilization
        cumprod_dim_parallel(tensor, dim)
    }
    
    #[cfg(not(feature = "std"))]
    {
        // Fallback to sequential implementation  
        let axis = Axis(dim);
        let mut array = tensor.array.into_owned();
        array.accumulate_axis_inplace(axis, |&prev, curr| *curr = *curr * prev);
        NdArrayTensor::new(array.into_shared())
    }
}

/// Cumulative maximum along axis using ndarray's accumulate_axis_inplace
pub(crate) fn cummax_dim<E: NdArrayElement + PartialOrd>(
    tensor: NdArrayTensor<E>,
    dim: usize,
) -> NdArrayTensor<E> {
    // TODO: Implement parallel cummax using burn_common::{run_par, iter_par} 
    // similar to cumsum_dim_parallel and cumprod_dim_parallel in scan_parallel.rs
    // for multi-core performance on large arrays
    let axis = Axis(dim);
    let mut array = tensor.array.into_owned();
    array.accumulate_axis_inplace(axis, |&prev, curr| {
        if prev > *curr {
            *curr = prev;
        }
    });
    NdArrayTensor::new(array.into_shared())
}

/// Cumulative minimum along axis using ndarray's accumulate_axis_inplace
pub(crate) fn cummin_dim<E: NdArrayElement + PartialOrd>(
    tensor: NdArrayTensor<E>,
    dim: usize,
) -> NdArrayTensor<E> {
    // TODO: Implement parallel cummin using burn_common::{run_par, iter_par}
    // similar to cumsum_dim_parallel and cumprod_dim_parallel in scan_parallel.rs
    // for multi-core performance on large arrays
    let axis = Axis(dim);
    let mut array = tensor.array.into_owned();
    array.accumulate_axis_inplace(axis, |&prev, curr| {
        if prev < *curr {
            *curr = prev;
        }
    });
    NdArrayTensor::new(array.into_shared())
}
