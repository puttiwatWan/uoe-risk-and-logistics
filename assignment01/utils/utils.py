from datetime import datetime as dt


def time_spent_decorator(func):
    """
    The function is a decorator which prints out when a function func is started
    and prints out the total time used to run the function once it is done.
    <br><br>

    :param func: A function to be decorated
    :returns: A wrapped function
    """

    def wrapper(*args, **kwargs):
        title = f"==> Starting function {func.__name__}"
        print(title)
        st = dt.now()
        func(*args, **kwargs)
        time_used_text = f"-- Total time used for function {func.__name__}: {(dt.now() - st).total_seconds()} seconds"
        print(time_used_text)
        print('-' * len(time_used_text))

    return wrapper
