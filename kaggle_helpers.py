################################################################################
## Anthony Lee
## 2024-11-22
##
## Copyright 2024 Anthony Lee
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the “Software”), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
################################################################################


def check_local_or_kaggle():
    """Helper function to check if running locally or on Kaggle."""
    import os
    from pathlib import Path

    try:
        USER = os.environ["USER"]
        print(f"Running locally as USER={USER}")
        globals()["RUNNING_ENVIRONMENT"] = "local"
    except KeyError as err_msg:
        try:
            KAGGLE_GCP_ZONE = os.environ["KAGGLE_GCP_ZONE"]
            KAGGLE_KERNEL_RUN_TYPE = os.environ["KAGGLE_KERNEL_RUN_TYPE"]

            globals()["DATA_SOURCE"] = Path("/kaggle/input/")
            globals()["WORKING_DIR"] = Path("/kaggle/working/")
            globals()["RUNNING_ENVIRONMENT"] = "kaggle"
            globals()["KAGGLE_KERNEL_RUN_TYPE"] = KAGGLE_KERNEL_RUN_TYPE

            print(
                f"Running on Kaggle GCP zone: {KAGGLE_GCP_ZONE} {KAGGLE_KERNEL_RUN_TYPE} mode"
            )
        except KeyError:
            print("Uhhh, something is wrong.")


def restart_kernel_if_kaggle_interactive():
    """Restarts the kernel if running in Kaggle interactively."""
    try:
        if globals()["KAGGLE_KERNEL_RUN_TYPE"] == "Interactive":
            import os
            from time import sleep
            from IPython.display import clear_output

            clear_output()
            sleep(3)
            os._exit(00)
    except KeyError:
        # Excepted error
        pass
