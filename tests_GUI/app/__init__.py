from flask import Flask

app = Flask(__name__)

from app import routes
from app import first_page
from app import second
from app import last_msg