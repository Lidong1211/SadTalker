# _*_coding:utf-8_*_
import threading
from flask import Flask


def _init_task(app):
    """
    开启相关任务
    """
    # 启动删除历史过期数据定时任务
    from app.tasks import delete_old_folders_task
    delete_old_folders_thread = threading.Thread(target=delete_old_folders_task, args=(app,))
    delete_old_folders_thread.start()
    # 启动初始化历史未完成任务
    from app.tasks import loading_old_task
    loading_old_task()
    # 启动生成视频任务
    from app.tasks import generate_video_task
    thread = threading.Thread(target=generate_video_task)
    thread.start()


def _init_blueprint(app):
    """
    # 注册蓝图
    """
    from app.views import video as video_blueprint
    # url_prefix 既可以放在定义，也可以放在注册时候，非常方便
    app.register_blueprint(video_blueprint, url_prefix='/api/video')


def create_app():
    """
    服务启动时
    """
    app = Flask(__name__)

    # 初始化定时任务
    _init_task(app)

    # 初始化蓝图
    _init_blueprint(app)

    return app
