import pickle
import hashlib
from pathlib import Path
from inspect import signature, getsource
from typing import Any, Callable
from functools import wraps

def hash_object(obj: Any) -> str:
    """
    Returns an MD5 hash of a pickled Python object.
    Args:
        obj: The object to hash.
    Returns:
        A string representing the MD5 hash.
    """
    return hashlib.md5(pickle.dumps(obj)).hexdigest()

def hash_func(func: Callable) -> str:
    """
    Returns an MD5 hash of a function's source code or its representation.
    Args:
        func: The function to hash.
    Returns:
        A string representing the MD5 hash of the function.
    """
    try:
        source = getsource(func)
    except OSError:
        source = repr(func)
    return hash_object(source)

def disk_cache(RESULT_CACHE: Path = Path("result_cache")) -> Callable:
    """
    A decorator that caches function results on disk using hashed arguments and function source.
    Args:
        RESULT_CACHE: Directory path to store cache files.
    Returns:
        A decorator that caches the results of the decorated function.
    """
    RESULT_CACHE.mkdir(exist_ok=True, parents=True)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            sig = signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            arg_hashes = [(name, hash_object(val)) for name, val in bound.arguments.items()]
            combined_key = hash_object((hash_func(func), arg_hashes))
            cache_path = RESULT_CACHE / f"{combined_key}.pkl"

            if cache_path.exists():
                with open(cache_path, "rb") as file:
                    return pickle.load(file)

            result = func(*args, **kwargs)
            with open(cache_path, "wb") as file:
                pickle.dump(result, file)
            return result

        return wrapper
    return decorator