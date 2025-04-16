import asyncio, logging, functools

def async_retry(max_retries: int=3, delay: int=1):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    logging.exception(f"{func.__name__} Attempt {attempt} failed: {str(e)}")
                    await asyncio.sleep(delay)
            raise ValueError(f"{func.__name__} Failed after {max_retries} attempts")
        return wrapper
    return decorator