import sys
import io
import pathlib
import traceback
import copy
import ast
import contextlib
import multiprocessing
from config import logging_config
import logging

from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from colorama import Fore, Style

class CodeExecutor:
    """
    A code execution environment for CodeAct agents with state management,
    output capture, error handling, and sandbox capabilities.
    """
    
    def __init__(
        self,
        predefined_functions: Optional[Dict[str, Callable]] = None,
        allowed_imports: Optional[List[str]] = None,
        allowed_filesystem_paths: Optional[List[str]] = None,
        allowed_file_modes: Optional[List[str]] = None,
        timeout: int = 30,
        max_output_size: int = 10000,
        use_multiprocessing: bool = False
    ):
        """
        Initialize the code executor.
        
        Args:
            predefined_functions: Dictionary of function_name -> function to make available
            allowed_imports: List of allowed module names (if None, all imports allowed)
            allowed_filesystem_paths: List of absolute directory paths where file operations are allowed. If None, no File operation is allowed.
            allowed_file_modes: List of allowed file modes ('r', 'w', 'a', etc.). If None, all modes allowed.
            timeout: Maximum execution time in seconds
            max_output_size: Maximum size of captured output in characters
            use_multiprocessing: If True, run code in separate process for better isolation
        """
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.allowed_imports = allowed_imports
        self.use_multiprocessing = use_multiprocessing
        self.allowed_filesystem_paths = self.ensure_all_absolute(allowed_filesystem_paths)
        self.allowed_file_modes = allowed_file_modes

        # Initialize persistent state
        self.globals_dict = self._create_safe_globals()
        self.locals_dict = {}
        
        # Add predefined functions to the global namespace
        if predefined_functions:
            for name, func in predefined_functions.items():
                self.globals_dict[name] = func
        
        # Track execution history
        self.execution_history = []
        
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a safe global namespace with controlled builtins."""
        # Start with safe builtins
        safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'ascii': ascii,
            'bin': bin,
            'bool': bool,
            'bytes': bytes,
            'callable': callable,
            'chr': chr,
            'complex': complex,
            'dict': dict,
            'dir': dir,
            'divmod': divmod,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'format': format,
            'frozenset': frozenset,
            'getattr': getattr,
            'hasattr': hasattr,
            'hash': hash,
            'hex': hex,
            'id': id,
            'int': int,
            'isinstance': isinstance,
            'issubclass': issubclass,
            'iter': iter,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'next': next,
            'object': object,
            'oct': oct,
            'ord': ord,
            'pow': pow,
            'print': print,
            'property': property,
            'range': range,
            'repr': repr,
            'reversed': reversed,
            'round': round,
            'set': set,
            'setattr': setattr,
            'slice': slice,
            'sorted': sorted,
            'staticmethod': staticmethod,
            'str': str,
            'sum': sum,
            'super': super,
            'tuple': tuple,
            'type': type,
            'vars': vars,
            'zip': zip,
            '__import__': self._safe_import,
            'open': self._restricted_open,  # Restricted file access
        }
        
        return {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
            '__doc__': None,
        }
    
    def _safe_import(self, name, *args, **kwargs):
        """Controlled import function that checks against allowed imports."""
        if self.allowed_imports is not None:
            # Check if the module or any parent module is allowed
            module_parts = name.split('.')
            allowed = False
            for i in range(len(module_parts), 0, -1):
                parent = '.'.join(module_parts[:i])
                if parent in self.allowed_imports:
                    allowed = True
                    break
            if not allowed:
                raise ImportError(f"Import of '{name}' is not allowed")
        
        return __import__(name, *args, **kwargs)
    
    def ensure_all_absolute(self, paths: List[str])->None:
        """
        Raise ValueError if any path in `paths` is not an absolute path.
        """
        if not paths:
            return [] # nothing to check

        not_absolute = [p for p in paths if not Path(p).is_absolute()]
        if not_absolute:
            raise ValueError(
                "The following paths are not absolute: " + ", ".join(not_absolute)
            )
        return paths
        
    def _restricted_open(self, file, mode='r', *args, **kwargs):
        """
        Restricted version of open() that only allows access to specified paths.
        
        Args:
            file: Path to the file to open
            mode: File mode ('r', 'w', 'a', etc.)
            *args, **kwargs: Additional arguments to pass to open()
        
        Returns:
            File handle if allowed
            
        Raises:
            PermissionError: If file access is not allowed
        """
        # If no paths are allowed, deny all file operations
        if not self.allowed_filesystem_paths:
            raise PermissionError("File operations are not allowed in this environment")
        
        # Resolve symlinks
        try:
            requested_path = pathlib.Path(file).resolve()
        except Exception as e:
            raise PermissionError(f"Invalid file path: {file}") from e
        
        # Check if file mode is allowed
        if self.allowed_file_modes is not None:
            # Extract base mode (remove '+' and 'b'/'t' modifiers)
            base_mode = mode.replace('+', '').replace('b', '').replace('t', '')
            if mode not in self.allowed_file_modes and base_mode not in self.allowed_file_modes:
                raise PermissionError(
                    f"File mode '{mode}' is not allowed. "
                    f"Allowed modes: {', '.join(self.allowed_file_modes)}"
                )
        
        # Check if the path is within allowed directories
        path_allowed = False
        allowed_parent = None
        
        for allowed_path in self.allowed_filesystem_paths:
            try:
                # Check if requested path is within allowed directory
                requested_path.relative_to(pathlib.Path(allowed_path))
                path_allowed = True
                allowed_parent = allowed_path
                break
            except ValueError:
                # Not relative to this allowed path, continue checking
                continue
        
        if not path_allowed:
            raise PermissionError(
                f"Access to '{requested_path}' is not allowed. "
                f"Allowed directories: {', '.join(self.allowed_filesystem_paths)}"
            )
        
        # Perform the actual open operation
        return open(file, mode, *args, **kwargs)
    
    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute code in the sandbox environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            Dictionary containing:
                - success: Boolean indicating if execution succeeded
                - output: Captured stdout/stderr
                - result: The result value (if any)
                - error: Error message if execution failed
                - traceback: Full traceback if error occurred
        """
        if self.use_multiprocessing:
            return self._execute_in_process(code)
        else:
            return self._execute_in_thread(code)
    
    def _execute_in_thread(self, code: str) -> Dict[str, Any]:
        """Execute code in the current thread with output capture."""
        # Prepare output capture
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        result = None
        error_msg = None
        tb_str = None
        
        # Create a context manager to capture output
        @contextlib.contextmanager
        def capture_output():
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        
        try:
            # Parse the code to check for syntax errors first
            ast.parse(code)
            
            # Execute with output capture
            with capture_output():
                # Try to evaluate as expression first (to capture return value)
                try:
                    result = eval(code, self.globals_dict, self.locals_dict)
                except SyntaxError:
                    # If not an expression, execute as statements
                    exec(code, self.globals_dict, self.locals_dict)
                    result = None
            
            # Get captured output
            output = stdout_capture.getvalue() + stderr_capture.getvalue()
            
            # Truncate output if too large
            if len(output) > self.max_output_size:
                output = output[:self.max_output_size] + "\n... (output truncated)"
            
            # Record successful execution
            execution_record = {
                'success': True,
                'output': output,
                'result': result,
                'error': None,
                'traceback': None
            }
            
        except Exception as e:
            # Capture error details
            error_msg = str(e)
            tb_str = traceback.format_exc()
            
            # Get any partial output
            output = stdout_capture.getvalue() + stderr_capture.getvalue()
            
            execution_record = {
                'success': False,
                'output': output,
                'result': None,
                'error': error_msg,
                'traceback': tb_str
            }
        
        # Add to history
        self.execution_history.append({
            'code': code,
            'result': execution_record
        })
        
        return execution_record
    
    def _execute_in_process(self, code: str) -> Dict[str, Any]:
        """Execute code in a separate process for better isolation."""
        def worker(code, globals_dict, locals_dict, queue):
            """Worker function for multiprocessing execution."""
            try:
                # Redirect output
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
                
                try:
                    # Execute the code
                    result = None
                    try:
                        result = eval(code, globals_dict, locals_dict)
                    except SyntaxError:
                        exec(code, globals_dict, locals_dict)
                    
                    # Restore output
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    
                    output = stdout_capture.getvalue() + stderr_capture.getvalue()
                    
                    # Send results back
                    queue.put({
                        'success': True,
                        'output': output,
                        'result': result,
                        'error': None,
                        'traceback': None,
                        'globals': globals_dict,
                        'locals': locals_dict
                    })
                    
                except Exception as e:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    
                    output = stdout_capture.getvalue() + stderr_capture.getvalue()
                    queue.put({
                        'success': False,
                        'output': output,
                        'result': None,
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                        'globals': globals_dict,
                        'locals': locals_dict
                    })
                    
            except Exception as e:
                queue.put({
                    'success': False,
                    'output': '',
                    'result': None,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'globals': {},
                    'locals': {}
                })
        
        # Create queue for communication
        queue = multiprocessing.Queue()
        
        # Create and start process
        process = multiprocessing.Process(
            target=worker,
            args=(code, copy.deepcopy(self.globals_dict), 
                  copy.deepcopy(self.locals_dict), queue)
        )
        process.start()
        
        # Wait for completion with timeout
        process.join(timeout=self.timeout)
        
        if process.is_alive():
            # Timeout occurred
            process.terminate()
            process.join()
            return {
                'success': False,
                'output': '',
                'result': None,
                'error': f'Execution timed out after {self.timeout} seconds',
                'traceback': None
            }
        
        # Get results
        try:
            result = queue.get_nowait()
            
            # Update state with changes from subprocess
            if result['success']:
                self.globals_dict.update(result.get('globals', {}))
                self.locals_dict.update(result.get('locals', {}))
            
            # Remove state from result
            result.pop('globals', None)
            result.pop('locals', None)
            
            return result
            
        except:
            return {
                'success': False,
                'output': '',
                'result': None,
                'error': 'Failed to retrieve execution results',
                'traceback': traceback.format_exc()
            }
    
    def reset_state(self):
        """Reset the execution environment to initial state."""
        self.globals_dict = self._create_safe_globals()
        self.locals_dict = {}
        self.execution_history = []
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of globals and locals."""
        return {
            'globals': {k: v for k, v in self.globals_dict.items() 
                       if not k.startswith('__')},
            'locals': self.locals_dict.copy()
        }
    
    def set_variable(self, name: str, value: Any):
        """Set a variable in the global namespace."""
        self.globals_dict[name] = value
    
    def get_variable(self, name: str, default=None):
        """Get a variable from the execution environment."""
        if name in self.locals_dict:
            return self.locals_dict[name]
        elif name in self.globals_dict:
            return self.globals_dict[name]
        return default
    
    def add_function(self, name: str, func: Callable):
        """Add a function to the global namespace."""
        self.globals_dict[name] = func
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the execution history."""
        return self.execution_history.copy()

if __name__== "__main__":
    logging.info(f"{Fore.BLUE}{Style.BRIGHT}[{__name__}] Running Code Executor for sample python code{Style.RESET_ALL}")
    sample_code = """from datetime import datetime
from colorama import Fore, Style
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
print(f"{Fore.GREEN}{Style.BRIGHT}[Code Executor] {Fore.BLACK}Current Time: {dt_string}{Style.RESET_ALL}")
"""
    executor = CodeExecutor()
    output = executor.execute(sample_code)
    logging.info(output['output'])
