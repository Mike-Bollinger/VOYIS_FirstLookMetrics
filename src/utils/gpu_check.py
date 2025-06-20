"""
GPU Check Utility

This script checks if GPU acceleration is available for TensorFlow operations.
Run this utility directly to check your system's GPU capabilities.
"""

import os
import sys
import traceback
import time

def check_gpu():
    """
    Check if GPU acceleration is available for TensorFlow and return detailed information
    
    Returns:
        dict: Dictionary containing GPU information and status
    """
    # Initialize result dictionary
    gpu_info = {
        'tensorflow_available': False,
        'tensorflow_version': None,
        'cuda_available': False,
        'cuda_version': None,
        'cudnn_version': None,
        'gpus_found': 0,
        'gpu_devices': [],
        'gpu_acceleration_available': False,
        'memory_limit': None,
        'error': None
    }
    
    print("Checking for GPU acceleration capabilities...")
    print("=" * 50)
    
    try:
        # Try to suppress TensorFlow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        # Check if TensorFlow is available
        print("1. Checking for TensorFlow installation...")
        try:
            import tensorflow as tf
            gpu_info['tensorflow_available'] = True
            gpu_info['tensorflow_version'] = tf.__version__
            print(f"   ✓ TensorFlow {tf.__version__} is installed")
        except ImportError:
            print("   ✗ TensorFlow is not installed")
            print("     To install: pip install tensorflow")
            return gpu_info
        except Exception as e:
            print(f"   ✗ Error importing TensorFlow: {e}")
            gpu_info['error'] = f"TensorFlow import error: {str(e)}"
            return gpu_info
            
        # Check CUDA availability
        print("\n2. Checking CUDA availability...")
        if hasattr(tf.sysconfig, 'get_build_info'):
            build_info = tf.sysconfig.get_build_info()
            if 'cuda_version' in build_info:
                gpu_info['cuda_version'] = build_info['cuda_version']
                gpu_info['cuda_available'] = True
                print(f"   ✓ CUDA {build_info['cuda_version']} is available")
                
                if 'cudnn_version' in build_info:
                    gpu_info['cudnn_version'] = build_info['cudnn_version']
                    print(f"   ✓ cuDNN {build_info['cudnn_version']} is available")
            else:
                print("   ✗ CUDA is not available in this TensorFlow build")
                print("     This might be a CPU-only version of TensorFlow")
        else:
            # Older TensorFlow versions don't have get_build_info
            if tf.test.is_built_with_cuda():
                gpu_info['cuda_available'] = True
                print("   ✓ CUDA is available (version information unavailable)")
            else:
                print("   ✗ CUDA is not available in this TensorFlow build")
        
        # Check for GPU devices
        print("\n3. Checking for GPU devices...")
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            gpu_info['gpus_found'] = len(physical_devices)
            
            if physical_devices:
                gpu_info['gpu_acceleration_available'] = True
                print(f"   ✓ Found {len(physical_devices)} GPU device(s):")
                
                for i, device in enumerate(physical_devices):
                    device_name = device.name.decode('utf-8') if hasattr(device.name, 'decode') else device.name
                    
                    # Try to get more detailed information about the GPU
                    try:
                        gpu_info['gpu_devices'].append(device_name)
                        print(f"     {i+1}. {device_name}")
                        
                        # Try to get device details using experimental API
                        try:
                            # Configure memory growth to avoid taking all GPU memory
                            tf.config.experimental.set_memory_growth(device, True)
                            print(f"        Memory growth enabled")
                        except Exception as e:
                            print(f"        Could not enable memory growth: {e}")
                    except Exception as detail_error:
                        print(f"     {i+1}. GPU device (details unavailable: {detail_error})")
                
                # Additional GPU test - try a simple operation
                print("\n4. Testing GPU with a simple computation...")
                try:
                    # Create and run a simple operation on GPU
                    with tf.device('/GPU:0'):
                        start_time = time.time()
                        # Create a large random tensor
                        x = tf.random.normal([5000, 5000])
                        # Perform matrix multiplication
                        result = tf.matmul(x, x)
                        # Force execution and synchronization
                        sum_result = tf.reduce_sum(result).numpy()
                        end_time = time.time()
                    
                    cpu_start_time = time.time()
                    with tf.device('/CPU:0'):
                        # Create a large random tensor
                        x_cpu = tf.random.normal([5000, 5000])
                        # Perform matrix multiplication
                        result_cpu = tf.matmul(x_cpu, x_cpu)
                        # Force execution and synchronization
                        sum_result_cpu = tf.reduce_sum(result_cpu).numpy()
                    cpu_end_time = time.time()
                    
                    gpu_time = end_time - start_time
                    cpu_time = cpu_end_time - cpu_start_time
                    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                    
                    print(f"   ✓ GPU computation successful")
                    print(f"     GPU time: {gpu_time:.4f} seconds")
                    print(f"     CPU time: {cpu_time:.4f} seconds")
                    print(f"     Speedup: {speedup:.2f}x")
                    
                    gpu_info['performance_test'] = {
                        'gpu_time': gpu_time,
                        'cpu_time': cpu_time,
                        'speedup': speedup
                    }
                    
                except Exception as test_error:
                    print(f"   ✗ GPU computation test failed: {test_error}")
                    print("     This may indicate a problem with the GPU configuration")
            else:
                print("   ✗ No GPU devices found")
                
            # Check memory limitations
            print("\n5. Checking TensorFlow memory configuration...")
            try:
                # Get memory info
                memory_info = []
                for device in tf.config.list_physical_devices('GPU'):
                    try:
                        memory_limit = tf.config.experimental.get_memory_info(device)
                        memory_info.append(memory_limit)
                    except:
                        pass
                
                if memory_info:
                    gpu_info['memory_limit'] = memory_info
                    print(f"   ✓ Memory configuration available: {memory_info}")
                else:
                    print("   ℹ Memory configuration information unavailable")
            except Exception as mem_error:
                print(f"   ℹ Could not get memory configuration: {mem_error}")
            
        except Exception as e:
            print(f"   ✗ Error checking GPU devices: {e}")
            gpu_info['error'] = f"GPU check error: {str(e)}"
        
        # Summarize findings
        print("\n" + "=" * 50)
        print("GPU Acceleration Summary:")
        if gpu_info['gpu_acceleration_available']:
            print(f"✓ GPU acceleration is AVAILABLE with {gpu_info['gpus_found']} device(s)")
            print(f"✓ TensorFlow {gpu_info['tensorflow_version']} is using CUDA")
            if hasattr(gpu_info, 'performance_test') and gpu_info['performance_test']['speedup'] > 1:
                print(f"✓ Performance test shows {gpu_info['performance_test']['speedup']:.2f}x speedup with GPU")
            print("\nImage visibility analysis will use GPU acceleration automatically.")
        else:
            print("✗ GPU acceleration is NOT AVAILABLE")
            if not gpu_info['tensorflow_available']:
                print("  - TensorFlow is not installed")
            elif not gpu_info['cuda_available']:
                print("  - CUDA is not available in the installed TensorFlow")
            elif gpu_info['gpus_found'] == 0:
                print("  - No compatible GPU devices were found")
            print("\nImage visibility analysis will use CPU processing automatically.")
            
    except Exception as e:
        print(f"Error during GPU check: {e}")
        traceback.print_exc()
        gpu_info['error'] = f"General error: {str(e)}"
    
    return gpu_info

def main():
    """Run the GPU check and display results when script is run directly"""
    print("\nVOYIS First Look Metrics - GPU Check Utility")
    print("==========================================\n")
    
    # Add parent directories to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
    # Run the GPU check
    gpu_info = check_gpu()
    
    # Print conclusions and recommendations
    print("\nConclusions and Recommendations:")
    print("------------------------------")
    
    if gpu_info['gpu_acceleration_available']:
        print("✓ Your system is configured correctly for GPU acceleration.")
        print("✓ The visibility analysis will automatically use your GPU.")
        
        # Check for potential optimizations
        if 'performance_test' in gpu_info and gpu_info['performance_test']['speedup'] < 5:
            print("\nℹ Note: Your GPU provides some acceleration, but the speedup is modest.")
            print("  This could be due to an older GPU model or driver configuration.")
            print("  The application will still use GPU acceleration when possible.")
    else:
        print("ℹ Your system will use CPU processing for visibility analysis.")
        print("  This is still fully functional but may be slower for large datasets.")
        
        # Provide specific recommendations based on the issue
        if not gpu_info['tensorflow_available']:
            print("\nTo enable GPU acceleration:")
            print("1. Install TensorFlow with GPU support:")
            print("   pip install tensorflow")
        elif not gpu_info['cuda_available']:
            print("\nTo enable GPU acceleration:")
            print("1. Make sure NVIDIA CUDA Toolkit is installed")
            print("2. Make sure NVIDIA cuDNN is installed")
            print("3. Install TensorFlow with GPU support:")
            print("   pip install tensorflow")
        elif gpu_info['gpus_found'] == 0:
            print("\nNo compatible NVIDIA GPU was detected. If you have a GPU:")
            print("1. Make sure your GPU drivers are up to date")
            print("2. Make sure NVIDIA CUDA Toolkit is installed and compatible with your GPU")
            print("3. Make sure your GPU is supported by TensorFlow")
    
    print("\nFor more information, visit: https://www.tensorflow.org/install/gpu")

if __name__ == "__main__":
    main()