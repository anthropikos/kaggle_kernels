def kaggle_setup():
    """Convenient Kaggle environment setup function.

    Clones the source from my repository and then manipulate the environment variable
    such that packages can be loaded.
    """
    import os
    import sys
    import subprocess
    from pathlib import Path


    if Path("/kaggle/working/kaggle_kernels").exists(): 
        subprocess.call("cd /kaggle/working/kaggle_kernels; git pull", shell=True)
        print("git pulled")
    else:
        subprocess.call("git clone https://github.com/anthropikos/kaggle_kernels.git", shell=True)
        print("git cloned")
    subprocess.call("cd /kaggle/working/kaggle_kernels", shell=True)

    package_dir_path = Path("/kaggle/working/kaggle_kernels/gan/src")

    package_dir_path = os.path.abspath(package_dir_path)
    
    if package_dir_path not in sys.path:
        sys.path.append(package_dir_path)

    return
