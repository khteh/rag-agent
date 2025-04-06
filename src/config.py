import os, json, logging
from dotenv import load_dotenv
from urllib import parse
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
    DEBUG = False
    TESTING = False
    SECRET_KEY:str = None
    SQLALCHEMY_DATABASE_URI:str = None
    POSTGRESQL_DATABASE_URI:str = None
    JWT_SECRET_KEY:str = None
    CHROMA_URI:str = None
    CHROMA_TOKEN:str = None
    NEO4J_USERNAME:str = None
    NEO4J_PASSWORD:str = None
    NEO4J_URI:str = None
    OLLAMA_URI:str = None
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self, environment="Development"):
        with open('/etc/ragagent_config.json', 'r') as f:
            config = json.load(f)
        self.SECRET_KEY = config["SECRET_KEY"] or "you-will-never-guess"
        self.SQLALCHEMY_DATABASE_URI = f"postgresql+psycopg://{os.environ.get('DB_USERNAME')}:{parse.quote(os.environ.get('DB_PASSWORD'))}@{config['DB_HOST']}/LangchainCheckpoint"
        self.POSTGRESQL_DATABASE_URI = f"postgresql://{os.environ.get('DB_USERNAME')}:{parse.quote(os.environ.get('DB_PASSWORD'))}@{config['DB_HOST']}/LangchainCheckpoint"
        self.JWT_SECRET_KEY = config["JWT_SECRET_KEY"]
        credential = os.environ.get('NEO4J_AUTH').split('/')
        self.CHROMA_URI = config["CHROMA_URI"]
        self.CHROMA_TOKEN = os.environ.get("CHROMA_TOKEN")
        self.NEO4J_USERNAME = credential[0]
        self.NEO4J_PASSWORD = credential[1]
        self.NEO4J_URI = config['NEO4J_URI']
        self.OLLAMA_URI = config['OLLAMA_URI']
        """
        https://docs.python.org/3/library/logging.html
        The level parameter now accepts a string representation of the level such as ‘INFO’ as an alternative to the integer constants such as INFO.
        """
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.basicConfig(filename='/var/log/ragagent/log', filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', level=config['LOGLEVEL'], datefmt='%Y-%m-%d %H:%M:%S')	

config = Config()