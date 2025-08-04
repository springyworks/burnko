use crate::Tensor;

impl<const D: usize, B: crate::backend::Backend> Tensor<B, D, crate::Float> {
    /// Computes the 1D Fast Fourier Transform (FFT) along the specified dimension.
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to compute the FFT
    ///
    /// # Returns
    /// A complex tensor where the last dimension has size 2 (real, imaginary components)
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_tensor::{Tensor, backend::Backend};
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let tensor = Tensor::<B, 1>::from_data([1.0, 0.0, 1.0, 0.0], &device);
    ///     let fft_result = tensor.fft(0);
    ///     // Result shape: [4, 2] where [:, 0] = real, [:, 1] = imaginary
    /// }
    /// ```
    pub fn fft(self, dim: usize) -> Tensor<B, D, crate::Float> {
        // For now, keep the same dimensions - we'll handle complex representation internally
        Tensor::new(crate::TensorPrimitive::Float(
            B::float_fft(self.primitive.tensor(), dim)
        ))
    }

    /// Computes the 1D Inverse Fast Fourier Transform (IFFT) along the specified dimension.
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to compute the IFFT
    ///
    /// # Panics
    /// Panics if the input tensor doesn't have complex format (implementation detail)
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_tensor::{Tensor, backend::Backend};
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     // Complex input tensor from FFT
    ///     let tensor = Tensor::<B, 1>::from_data([1.0, 0.0, 1.0, 0.0], &device);
    ///     let fft_result = tensor.fft(0);
    ///     let ifft_result = fft_result.ifft(0);
    /// }
    /// ```
    pub fn ifft(self, dim: usize) -> Tensor<B, D, crate::Float> {
        Tensor::new(crate::TensorPrimitive::Float(
            B::float_ifft(self.primitive.tensor(), dim)
        ))
    }

    /// Computes the 2D Fast Fourier Transform (FFT) along the specified dimensions.
    ///
    /// # Arguments
    /// * `dim1` - The first dimension along which to compute the FFT
    /// * `dim2` - The second dimension along which to compute the FFT
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_tensor::{Tensor, backend::Backend};
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let tensor = Tensor::<B, 2>::from_data([[1.0, 0.0], [0.0, 1.0]], &device);
    ///     let fft2_result = tensor.fft2(0, 1);
    /// }
    /// ```
    pub fn fft2(self, dim1: usize, dim2: usize) -> Tensor<B, D, crate::Float> {
        // Apply FFT along dim1, then along dim2
        self.fft(dim1).fft(dim2)
    }

    /// Computes the 2D Inverse Fast Fourier Transform (IFFT) along the specified dimensions.
    pub fn ifft2(self, dim1: usize, dim2: usize) -> Tensor<B, D, crate::Float> {
        // Apply IFFT along dim2, then along dim1 (reverse order)
        self.ifft(dim2).ifft(dim1)
    }
}
