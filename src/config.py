import os, json
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
    NEO4J_USERNAME:str = None
    NEO4J_PASSWORD:str = None
    NEO4J_URI:str = None
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
        self.NEO4J_USERNAME = credential[0]
        self.NEO4J_PASSWORD = credential[1]
        self.NEO4J_URI = config['NEO4J_URI']

config = Config()