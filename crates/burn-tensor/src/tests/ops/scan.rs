#[burn_tensor_testgen::testgen(scan)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_cumsum_1d() {
        let tensor = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0, 5.0]);

        let output = tensor.cumsum(0);

        output
            .into_data()
            .assert_eq(&TensorData::from([1.0, 3.0, 6.0, 10.0, 15.0]), false);
    }

    #[test]
    fn should_support_cumsum_2d_axis_0() {
        let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let output = tensor.cumsum(0);

        output
            .into_data()
            .assert_eq(&TensorData::from([[1.0, 2.0, 3.0], [5.0, 7.0, 9.0]]), false);
    }

    #[test]
    fn should_support_cumsum_2d_axis_1() {
        let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let output = tensor.cumsum(1);

        output
            .into_data()
            .assert_eq(&TensorData::from([[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]]), false);
    }

    #[test]
    fn should_support_cumsum_3d() {
        let tensor = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);

        let output = tensor.cumsum(0);

        output
            .into_data()
            .assert_eq(&TensorData::from([[[1.0, 2.0], [3.0, 4.0]], [[6.0, 8.0], [10.0, 12.0]]]), false);
    }

    #[test]
    fn should_support_cumprod_1d() {
        let tensor = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0]);

        let output = tensor.cumprod(0);

        output
            .into_data()
            .assert_eq(&TensorData::from([1.0, 2.0, 6.0, 24.0]), false);
    }

    #[test]
    fn should_support_cumprod_2d_axis_0() {
        let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]);

        let output = tensor.cumprod(0);

        output
            .into_data()
            .assert_eq(&TensorData::from([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]), false);
    }

    #[test]
    fn should_support_cumprod_2d_axis_1() {
        let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [2.0, 3.0, 2.0]]);

        let output = tensor.cumprod(1);

        output
            .into_data()
            .assert_eq(&TensorData::from([[1.0, 2.0, 6.0], [2.0, 6.0, 12.0]]), false);
    }

    #[test]
    fn should_support_cumsum_empty_tensor() {
        let tensor = TestTensor::<1>::from(TensorData::from([] as [f32; 0]));

        let output = tensor.cumsum(0);

        assert_eq!(output.dims(), [0]);
    }

    #[test]
    fn should_support_cumsum_single_element() {
        let tensor = TestTensor::<1>::from([42.0]);

        let output = tensor.cumsum(0);

        output
            .into_data()
            .assert_eq(&TensorData::from([42.0]), false);
    }

    #[test]
    fn should_support_cumsum_negative_values() {
        let tensor = TestTensor::<1>::from([-1.0, 2.0, -3.0, 4.0]);

        let output = tensor.cumsum(0);

        output
            .into_data()
            .assert_eq(&TensorData::from([-1.0, 1.0, -2.0, 2.0]), false);
    }

    #[test]
    fn should_support_cumprod_with_zeros() {
        let tensor = TestTensor::<1>::from([1.0, 0.0, 3.0, 4.0]);

        let output = tensor.cumprod(0);

        output
            .into_data()
            .assert_eq(&TensorData::from([1.0, 0.0, 0.0, 0.0]), false);
    }

    #[test]
    fn should_support_cumsum_large_values() {
        let tensor = TestTensor::<1>::from([1e6, 2e6, 3e6]);

        let output = tensor.cumsum(0);

        output
            .into_data()
            .assert_eq(&TensorData::from([1e6, 3e6, 6e6]), false);
    }
}