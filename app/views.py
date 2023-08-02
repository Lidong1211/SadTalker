# coding:utf-8

"""
video 模块视图

"""
import os
from flask import Blueprint, request, jsonify, send_file

import config
from app.args_utils import build_args
from app.tasks import add_task
from app.tasks import index_of_task_id_list

video = Blueprint("video", __name__)


@video.route('/build', methods=['POST'])
def build_video():
    # 验证POST请求的数据
    if 'driven_audio' not in request.files:
        return jsonify({'error': 'driven_audio can not be empty'}), 400
    if 'source_image' not in request.files:
        return jsonify({'error': 'source_image can not be empty'}), 400
    # 格式化POST数据
    args = build_args(request)
    # 提交进入队列
    add_task(args)
    # 返回处理结果
    return jsonify({'task_id': args.task_id})


@video.route('/get', methods=['GET'])
def get_video():
    task_id = request.args.get('task_id')

    index, list_len = index_of_task_id_list(task_id)
    if index == 0:
        return jsonify({'status': 'processing', 'index': index + 1, "total": list_len})
    elif index != -1:
        return jsonify({'status': 'waiting', 'index': index + 1, "total": list_len})

    video_path = os.path.join(config.TMP_PATH, task_id, "result.mp4")
    if os.path.isfile(video_path):
        # 使用 send_file 函数返回视频文件
        return send_file(video_path, as_attachment=True)

    if os.path.isdir(os.path.join(config.TMP_PATH, task_id)):
        # 抛出异常，文件夹存在但文件不存在的情况
        return jsonify({'error': task_id + " video build exception"})
    # 抛出异常, taskId不存在异常
    return jsonify({'error': task_id + " is not exist"})
