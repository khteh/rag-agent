import asyncio, logging, functools

"""
https://realpython.com/primer-on-python-decorators/#defining-decorators-with-arguments
(1) Defining wrapper() as an inner function means that async_retry() will refer to a function object, wrapper.
    Now, you need to add parentheses when setting up the decorator, as in @async_retry(). This is necessary in order to add arguments.
(2) The max_retries and delay arguments are seemingly not used in async_retry() itself. But by passing max_retries and delay, a closure is created where the value of max_retries and delay are stored until wrapper() uses it later.
"""
def async_retry(max_retries: int=3, delay: int=0):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logging.exception(f"{func.__name__} Attempt {attempt} failed: {str(e)}")
                    await asyncio.sleep(delay)
            raise ValueError(f"{func.__name__} Failed after {max_retries} attempts")
        return wrapper
    return decorator