# -*-coding:utf-8-*-
from dotenv import load_dotenv
import os

load_dotenv()

# HOST
SERVER_HOST = os.getenv('SERVER_HOST')
# 端口号
SERVER_PORT = os.getenv('SERVER_PORT')
# 文件保存目录
TMP_PATH = os.getenv('TMP_PATH')
