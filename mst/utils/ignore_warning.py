import warnings


def suppress_mst_warnings(all_warnings=False):
    if all_warnings:
        warnings.filterwarnings("ignore")
        return

    warnings.filterwarnings("ignore", message="xFormers is not available")
    warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
    warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")
    warnings.filterwarnings("ignore", message="A module that was compiled using NumPy 1.x")
    warnings.filterwarnings("ignore", message=".*_ARRAY_API not found.*")