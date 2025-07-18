// Import the shared macro
use crate::include_models;
include_models!(expand, expand_tensor, expand_shape);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Shape, Tensor};

    use crate::backend::Backend;

    #[test]
    fn expand() {
        let device = Default::default();
        let model: expand::Model<Backend> = expand::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[-1.0], [1.0]], &device);

        let output = model.forward(input1);
        let expected_shape = Shape::from([2, 2]);

        assert_eq!(output.shape(), expected_shape);
    }

    #[test]
    fn expand_tensor() {
        let device = Default::default();
        let model: expand_tensor::Model<Backend> = expand_tensor::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[-1.0], [1.0]], &device);
        let input2 = Tensor::<Backend, 1, Int>::from_ints([2, 2], &device);

        let output = model.forward(input1, input2);
        let expected_shape = Shape::from([2, 2]);

        assert_eq!(output.shape(), expected_shape);
    }

    #[test]
    fn expand_shape() {
        let device = Default::default();
        let model: expand_shape::Model<Backend> = expand_shape::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[-1.0], [1.0], [1.0], [1.0]], &device);
        let input2 = Tensor::<Backend, 2>::zeros([4, 4], &device);

        let output = model.forward(input1, input2);
        let expected_shape = Shape::from([4, 4]);

        assert_eq!(output.shape(), expected_shape);
    }
}
