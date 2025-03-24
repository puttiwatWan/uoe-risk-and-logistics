from datetime import datetime as dt


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

        title = f"==> Starting function {func.__name__}"
        print(title)
        st = dt.now()
        result = func(*args, **kwargs)  # Capture the result
        time_used_text = f"-- Total time used for function {func.__name__}: {(dt.now() - st).total_seconds()} seconds"
        print(time_used_text)
        print('-' * len(time_used_text))
        return result  # Return the result to the caller
    return wrapper
