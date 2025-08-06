// Test to understand ndarray's built-in accumulate_axis_inplace behavior

use ndarray::{Array2, Axis};
use std::time::Instant;

#[test]
fn test_ndarray_accumulate_performance() {
    println!("=== Testing ndarray's accumulate_axis_inplace ===");
    
    // Test 1: Correctness on small array
    let small_array = Array2::from_shape_vec((2, 3), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    println!("Original: {:?}", small_array);
    
    let mut test_axis0 = small_array.clone();
    test_axis0.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr = *curr + prev);
    println!("After accumulate axis 0: {:?}", test_axis0);
    
    let mut test_axis1 = small_array.clone();
    test_axis1.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr = *curr + prev);
    println!("After accumulate axis 1: {:?}", test_axis1);
    
    // Test 2: Performance test on large array
    println!("\n=== Performance Test ===");
    let size = 1000;
    let large_array = Array2::from_elem((size, size), 1.0f32);
    
    // Test axis 0 performance
    let mut test_large_axis0 = large_array.clone();
    let start = Instant::now();
    test_large_axis0.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr = *curr + prev);
    let duration_axis0 = start.elapsed();
    println!("Axis 0 cumsum on {}x{} took: {:?}", size, size, duration_axis0);
    
    // Test axis 1 performance
    let mut test_large_axis1 = large_array.clone();
    let start = Instant::now();
    test_large_axis1.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr = *curr + prev);
    let duration_axis1 = start.elapsed();
    println!("Axis 1 cumsum on {}x{} took: {:?}", size, size, duration_axis1);
    
    // Verify results make sense
    let final_corner = test_large_axis0[[size-1, size-1]];
    println!("Final corner value (axis 0): {} (expected: {})", final_corner, size as f32);
    
    let final_corner = test_large_axis1[[size-1, size-1]];
    println!("Final corner value (axis 1): {} (expected: {})", final_corner, size as f32);
    
    // Test 3: Check if ndarray has parallel features enabled
    println!("\n=== Feature Detection ===");
    #[cfg(feature = "std")]
    println!("Std feature is enabled - parallel operations available!");
    #[cfg(not(feature = "std"))]
    println!("Std feature is NOT enabled - sequential operations only");
}

#[test]
fn test_thread_usage() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    println!("=== Thread Usage Test ===");
    let thread_ids = Arc::new(Mutex::new(Vec::new()));
    
    // Create a 1000x1000 array
    let mut large_array = Array2::from_elem((1000, 1000), 1.0f32);
    
    // Monkey-patch to detect thread usage (this is a hack)
    let thread_ids_clone = thread_ids.clone();
    let custom_op = move |&prev: &f32, curr: &mut f32| {
        let current_thread = thread::current().id();
        if let Ok(mut ids) = thread_ids_clone.lock() {
            if !ids.contains(&current_thread) {
                ids.push(current_thread);
            }
        }
        *curr = *curr + prev;
    };
    
    large_array.accumulate_axis_inplace(Axis(0), custom_op);
    
    let thread_count = thread_ids.lock().unwrap().len();
    println!("Number of threads used: {}", thread_count);
    println!("Available parallelism: {}", thread::available_parallelism().unwrap().get());
}
