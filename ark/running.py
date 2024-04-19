import time


class Timer:
    def __init__(self, name=None):
        self.name = name if name is not None else "Timer"

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{self.name} taken by {func.__name__}: {end_time - start_time} seconds")
            return result

        return wrapper
