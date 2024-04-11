import traceback


def complete(func):
    def inner(*args, **kwargs):
        import_error = False
        try:
            from playsound import playsound
        except ImportError:
            import_error = True

        if import_error:
            func(*args, **kwargs)
        else:
            from playsound import playsound
            try:
                func(*args, **kwargs)
                playsound(r"complete.mp3")
            except RuntimeError as e:
                playsound(r"error.mp3")
                traceback.print_exc()
                print(e)

    return inner