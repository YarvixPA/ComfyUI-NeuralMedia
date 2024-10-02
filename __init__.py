import importlib.util
import subprocess
import sys
from pathlib import Path
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def load_module(module_name, file_path):
    """Loads a module from a .py file."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def install_requirements():
    """Installs dependencies if the requirements.txt file exists."""
    req_path = Path(__file__).parent / 'requirements.txt'
    if req_path.exists():
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(req_path)])
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}\n{traceback.format_exc()}")

def load_nodes():
    """Loads nodes from the nodes folder."""
    nodes_path = Path(__file__).parent / 'nodes'
    for file in nodes_path.glob("*.py"):
        if file.name != "__init__.py":
            try:
                module = load_module(f"nodes.{file.stem}", str(file))
                NODE_CLASS_MAPPINGS.update(getattr(module, 'NODE_CLASS_MAPPINGS', {}))
                NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, 'NODE_DISPLAY_NAME_MAPPINGS', {}))
            except Exception as e:
                print(f"Error loading {file.name}: {e}\n{traceback.format_exc()}")

# Execute functions
install_requirements()
load_nodes()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

