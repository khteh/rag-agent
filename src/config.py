import os, json, logging, sys
from logging.handlers import TimedRotatingFileHandler
from dotenv import load_dotenv
from urllib import parse
"""
https://pypi.org/project/python-dotenv/
By default, load_dotenv() will:

Look for a .env file in the same directory as the Python script (or higher up the directory tree).
Read each key-value pair and add it to os.environ.
Not override an environment variable that is already set, unless you explicitly pass override=True.
"""
load_dotenv()
class ConfigSingleton(type): # Inherit from "type" in order to gain access to method __call__
    __registry = {}
    def __call__(cls, *args, **kwargs):
        registry = type(cls).__registry
        if cls not in registry:
              registry[cls] = (super().__call__(*args, **kwargs), args, kwargs)
        elif registry[cls][1] != args or registry(cls)[2] != kwargs:
              raise TypeError(f"Class already initialized with different arguments!")
        return registry[cls][0]

class Config(metaclass=ConfigSingleton):
    TESTING = False
    LOGLEVEL:str = None
    SECRET_KEY:str = None
    SQLALCHEMY_DATABASE_URI:str = None
    POSTGRESQL_DATABASE_URI:str = None
    DB_MAX_CONNECTIONS:int = None
    JWT_SECRET_KEY:str = None
    #CHROMA_URI:str = None
    #CHROMA_TOKEN:str = None
    NEO4J_USERNAME:str = None
    NEO4J_PASSWORD:str = None
    NEO4J_URI:str = None
    BASE_URI:str = None
    GEMINI_API_KEY:str = None
    EMBEDDING_MODEL: str = None
    EMBEDDING_DIMENSIONS: int = None
    MODEL_PROVIDER: str = None
    LLM_RAG_MODEL: str = None
    connection_kwargs = None
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self, environment="Development"):
        with open('/etc/ragagent_config.json', 'r') as f:
            config = json.load(f)
        self.LOGLEVEL = config['LOGLEVEL']
        self.SECRET_KEY = config["SECRET_KEY"] or "you-will-never-guess"
        self.SQLALCHEMY_DATABASE_URI = f"postgresql+psycopg://{os.environ.get('DB_USERNAME')}:{parse.quote(os.environ.get('DB_PASSWORD'))}@{config['DB_HOST']}/Langchain"
        self.POSTGRESQL_DATABASE_URI = f"postgresql://{os.environ.get('DB_USERNAME')}:{parse.quote(os.environ.get('DB_PASSWORD'))}@{config['DB_HOST']}/Langchain"
        self.DB_MAX_CONNECTIONS = config["DB_MAX_CONNECTIONS"]
        self.JWT_SECRET_KEY = config["JWT_SECRET_KEY"]
        credential = os.environ.get('NEO4J_AUTH').split('/')
        #self.CHROMA_URI = config["CHROMA_URI"]
        #self.CHROMA_TOKEN = os.environ.get("CHROMA_TOKEN")
        self.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        self.NEO4J_USERNAME = credential[0]
        self.NEO4J_PASSWORD = credential[1]
        self.NEO4J_URI = config['NEO4J_URI']
        self.BASE_URI = config['BASE_URI']
        self.EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
        self.EMBEDDING_DIMENSIONS = config["EMBEDDING_DIMENSIONS"]
        self.MODEL_PROVIDER = config["MODEL_PROVIDER"]
        self.LLM_RAG_MODEL = config["LLM_RAG_MODEL"]
        self.connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        """
        https://docs.python.org/3/library/logging.html
        The level parameter now accepts a string representation of the level such as ‘INFO’ as an alternative to the integer constants such as INFO.
        httpx library is a dependency of LangGraph and is used under the hood to communicate with the AI models.
        """
        logging.getLogger("httpx").setLevel(logging.WARNING)
        """
        https://realpython.com/python-modulo-string-formatting/#fine-tune-your-output-with-conversion-flags
        -	Justification of values that are shorter than the specified field width
        The Hyphen-Minus Flag (-)
        When a formatted value is shorter than the specified field width, it’s usually right-justified in the field. The hyphen-minus (-) flag causes the value to be left-justified in the specified field instead.
        """
        if config["ENVIRONMENT"] == "development":
            logging.basicConfig(filename='/var/log/ragagent/log', filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', level=self.LOGLEVEL, datefmt='%Y-%m-%d %H:%M:%S')
        else:
            logging.basicConfig(handlers=[
                TimedRotatingFileHandler(filename='/var/log/ragagent/log', when='d', interval=1, backupCount=3),
                logging.StreamHandler(sys.stdout)
            ], format='%(asctime)s %(levelname)-8s %(message)s', level=self.LOGLEVEL, datefmt='%Y-%m-%d %H:%M:%S')

config = Config()