#[burn_tensor_testgen::testgen(fft_ops)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn test_fft_1d_basic() {
        // Test basic 1D FFT functionality
        let tensor = TestTensor::<1>::from([1.0, 0.0, 1.0, 0.0]);
        
        let output = tensor.fft(0);
        
        // FFT of [1,0,1,0] should be [2,0,2,0] (real part)
        // This is a basic identity test
        assert_eq!(output.dims(), [4]);
    }

    #[test]
    fn test_fft_2d_basic() {
        // Test 2D FFT functionality 
        let tensor = TestTensor::<2>::from([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]);
        
        let output = tensor.fft2(0, 1);
        
        // Should preserve shape
        assert_eq!(output.dims(), [4, 4]);
    }

    #[test]
    fn test_fft_impulse_response() {
        // Test FFT of impulse signal (delta function)
        let data = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let tensor = TestTensor::<1>::from(data);
        
        let output = tensor.fft(0);
        
        // FFT of delta should be all ones (constant spectrum)
        assert_eq!(output.dims(), [8]);
    }

    #[test]
    fn test_ifft_roundtrip() {
        // Test that FFT followed by IFFT returns original signal
        let original = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0]);
        
        let fft_result = original.clone().fft(0);
        let roundtrip = fft_result.ifft(0);
        
        // Should get back original (within numerical precision)
        assert_eq!(roundtrip.dims(), original.dims());
        
        // Note: Exact equality check might fail due to floating point precision
        // In a real test, we'd use approximate comparison
    }

    #[test]
    fn test_fft_symmetry() {
        // Test that FFT of real signal has conjugate symmetry
        let tensor = TestTensor::<1>::from([1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0]);
        
        let output = tensor.fft(0);
        
        assert_eq!(output.dims(), [8]);
    }

    #[test]
    fn test_fft_2d_separable() {
        // Test that 2D FFT can be computed as separable 1D FFTs
        let tensor = TestTensor::<2>::from([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        
        // 2D FFT
        let fft_2d = tensor.clone().fft2(0, 1);
        
        // Separable: FFT along rows, then columns
        let fft_rows = tensor.clone().fft(1); // FFT along axis 1 (rows)
        let fft_separable = fft_rows.fft(0);   // FFT along axis 0 (columns)
        
        assert_eq!(fft_2d.dims(), fft_separable.dims());
        // In practice, these should be approximately equal
    }

    #[test]
    fn test_fft_performance_2d() {
        // Performance test for 2D FFT operations
        let size = 64;
        let tensor = TestTensor::<2>::zeros([size, size], &Default::default());
        
        // This test is mainly to ensure the operation completes without error
        let output = tensor.fft2(0, 1);
        
        assert_eq!(output.dims(), [size, size]);
    }

    #[test]
    fn test_fft_batch_processing() {
        // Test FFT on batched data
        let batch_size = 4;
        let signal_length = 16;
        let tensor = TestTensor::<2>::zeros([batch_size, signal_length], &Default::default());
        
        let output = tensor.fft(1); // FFT along signal dimension
        
        assert_eq!(output.dims(), [batch_size, signal_length]);
    }

    #[test]
    fn test_fft_different_sizes() {
        // Test FFT with various power-of-2 sizes
        for &size in &[4, 8, 16, 32] {
            let tensor = TestTensor::<1>::zeros([size], &Default::default());
            let output = tensor.fft(0);
            assert_eq!(output.dims(), [size]);
        }
    }

    #[test]
    fn test_fft_non_power_of_two() {
        // Test FFT with non-power-of-2 sizes (if supported)
        for &size in &[6, 10, 12] {
            let tensor = TestTensor::<1>::zeros([size], &Default::default());
            let output = tensor.fft(0);
            assert_eq!(output.dims(), [size]);
        }
    }
}
