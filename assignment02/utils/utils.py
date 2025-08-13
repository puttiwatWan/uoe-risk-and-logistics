from datetime import datetime as dt
from typing import Union, List, Tuple

import numpy as np

print_time = True


def set_print_time(b: bool):
    global print_time
    print_time = b


def time_spent_decorator(func):
    """
    Decorator that prints when the function starts and its execution time.
    """
    def wrapper(*args, **kwargs):
        if not print_time:
            return func(*args, **kwargs)

        char = "="
        title = f"==> Starting function {func.__name__}"
        print(title)
        st = dt.now()
        result = func(*args, **kwargs)  # Capture the result
        time_used_text = f"{2 * char} Total time used for function {func.__name__}: {(dt.now() - st).total_seconds()} seconds"
        print(time_used_text)
        print(char * len(time_used_text))
        return result  # Return the result to the caller
    return wrapper


def print_separator(header: str = ""):
    __char = "-"
    __half_side_char_len = 20
    __space_wrap_header = 0
    if header:
        __space_wrap_header = 1
    __splitter_char_len = 2 * __half_side_char_len + len(header) + 2 * __space_wrap_header

    print()
    print(__char * __splitter_char_len)
    print(__char * __half_side_char_len + " " * __space_wrap_header +
          f"{header}" + " " * __space_wrap_header + __char * __half_side_char_len)
    print(__char * __splitter_char_len)


def find_centroid(locations: Union[np.ndarray | List[Union[List | Tuple]]]) -> Tuple:
    return tuple(np.mean(locations, axis=0))
