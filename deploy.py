# -*-coding:utf-8-*-
import config
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT)
