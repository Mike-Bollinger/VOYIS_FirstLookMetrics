import tkinter as tk
import os
import sys
import traceback
from typing import Optional

def setup_tensorflow() -> Optional[str]:
    """
    Configure TensorFlow based on available hardware
    Returns: Status message about TensorFlow configuration
    """
    try:
        import tensorflow as tf
        
        # Prevent TensorFlow from allocating all GPU memory
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"\nEnabled memory growth for GPU: {gpu}")
                except RuntimeError as e:
                    print(f"\nError configuring GPU {gpu}: {e}")
            
            return f"TensorFlow {tf.__version__} initialized with GPU support ({len(gpus)} GPU(s) available)"
        else:
            return f"TensorFlow {tf.__version__} initialized (CPU only)"
            
    except ImportError:
        return "TensorFlow not available - visibility analysis will be disabled"
    except Exception as e:
        return f"Error initializing TensorFlow: {str(e)}"

def show_error_window(root, error_message):
    """Display an error window with the given message"""
    try:
        error_window = tk.Toplevel(root)
        error_window.title("Error")
        error_window.geometry("600x400")
        
        error_text = tk.Text(error_window, wrap=tk.WORD)
        error_text.pack(fill=tk.BOTH, expand=True)
        error_text.insert(tk.END, error_message)
        
        tk.Button(error_window, text="Close", command=root.destroy).pack(pady=10)
    except Exception as e:
        print(f"Failed to show error window: {str(e)}")
        print(error_message)

def main():
    # Wrap everything in a try/except
    try:
        # Add the parent directory to the path to enable imports from sibling packages
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)
        
        # Set up exception hook to catch errors in threads
        def exception_hook(exc_type, exc_value, exc_traceback):
            error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            print(f"Unhandled exception: {error_msg}")
            
        sys.excepthook = exception_hook
        
        # Configure TensorFlow before creating the window
        tf_status = setup_tensorflow()
        print(f"\nTensorFlow Status: {tf_status}")
        
        # Create the root window before any imports that might use tkinter
        root = tk.Tk()
        root.title("VOYIS First Look Metrics")
        
        try:
            # Import the AppWindow class after adjusting the path
            from src.gui.app_window import AppWindow
            
            # Create the app instance
            app = AppWindow(root)
            
            # Start the main loop
            root.mainloop()
        except Exception as e:
            error_message = f"Error initializing application: {str(e)}\n\n{traceback.format_exc()}"
            print(error_message)
            
            if root and root.winfo_exists():
                show_error_window(root, error_message)
    except Exception as e:
        print(f"Critical error in main function: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()