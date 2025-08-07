use ndarray::*;

fn main() {
    // Test tensor: [[1, 2], [3, 4]]
    let mut arr = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    println!("Original array:");
    println!("{:?}", arr);
    println!("Shape: {:?}", arr.shape());
    
    // Test axis_iter along axis 0 
    println!("\naxis_iter_mut(Axis(0)) - iterates OVER rows:");
    for (i, row) in arr.axis_iter_mut(Axis(0)).enumerate() {
        println!("Row {}: {:?}", i, row);
    }
    
    println!("\naxis_iter_mut(Axis(1)) - iterates OVER columns:");
    let mut arr2 = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    for (i, col) in arr2.axis_iter_mut(Axis(1)).enumerate() {
        println!("Col {}: {:?}", i, col);
    }
    
    // Test accumulate_axis_inplace
    println!("\nTesting accumulate_axis_inplace:");
    let mut test1 = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    println!("Before accumulate axis 0: {:?}", test1);
    test1.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr = *curr + prev);
    println!("After accumulate axis 0: {:?}", test1);
    
    let mut test2 = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    println!("Before accumulate axis 1: {:?}", test2);
    test2.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr = *curr + prev);
    println!("After accumulate axis 1: {:?}", test2);
}
