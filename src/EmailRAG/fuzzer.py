import atheris, asyncio, sys
from src.EmailRAG.EmailRAG import main

@atheris.instrument_func
def FuzzEntryPoint(data):
    # Initialize the provider with raw bytes from the fuzzer
    fdp = atheris.FuzzedDataProvider(data)    
    # Consume structured data
    number = fdp.ConsumeIntInRange(100, 100000)
    criteria = fdp.ConsumeUnicodeNoSurrogates(128)
    asyncio.run(main(criteria))

if __name__ == "__main__":
    atheris.Setup(sys.argv, FuzzEntryPoint)
    atheris.instrument_all()    
    atheris.Fuzz()    
