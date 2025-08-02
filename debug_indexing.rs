fn main() {
    // 3x3 matrix in row-major order:
    // [1, 0, 0,  // row 0: flat indices 0, 1, 2
    //  0, 1, 0,  // row 1: flat indices 3, 4, 5  
    //  0, 0, 1]  // row 2: flat indices 6, 7, 8
    
    // For scanning along dimension 1 (rows):
    // - We have 3 scan lines (one per row)
    // - Scan line 0: flat indices [0, 1, 2] -> cumsum [1, 1, 1]
    // - Scan line 1: flat indices [3, 4, 5] -> cumsum [0, 1, 1]  
    // - Scan line 2: flat indices [6, 7, 8] -> cumsum [0, 0, 1]
    
    // For a 3x3 tensor with shape [3, 3]:
    // - input.shape(0) = 3, input.shape(1) = 3
    // - input.stride(0) = 3, input.stride(1) = 1
    // - scan_dim = 1, scan_dim_size = 3
    // - total_elements = 9, num_scan_lines = 9/3 = 3
    
    println!("Testing indexing logic for 3x3 matrix scanning along dim 1:");
    
    let shape = [3u32, 3u32];
    let stride = [3u32, 1u32];
    let scan_dim = 1u32;
    let scan_dim_size = shape[scan_dim as usize];
    let total_elements = 9u32;
    let num_scan_lines = total_elements / scan_dim_size;
    
    println!("scan_dim_size: {}, num_scan_lines: {}", scan_dim_size, num_scan_lines);
    
    // Test the indexing logic for each thread
    for thread_id in 0..num_scan_lines {
        println!("\nThread {}: ", thread_id);
        
        // My current algorithm
        let mut base_offset = 0u32;
        let mut temp_thread_id = thread_id;
        
        for dim in 0..shape.len() {
            if dim as u32 != scan_dim {
                let stride_val = stride[dim];
                let size = shape[dim];
                let idx = temp_thread_id % size;
                base_offset += idx * stride_val;
                temp_thread_id /= size;
                println!("  dim {}: size={}, stride={}, idx={}, base_offset={}", 
                         dim, size, stride_val, idx, base_offset);
            }
        }
        
        println!("  Final base_offset: {}", base_offset);
        
        // Expected scan line indices
        let expected_indices: Vec<u32> = (0..scan_dim_size)
            .map(|i| base_offset + i * stride[scan_dim as usize])
            .collect();
        println!("  Scan line indices: {:?}", expected_indices);
    }
}
