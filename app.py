import uuid
import os, sys, time
import torch
import queue
import threading
import types
import shutil
from flask import Flask, jsonify, request, send_file
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

app = Flask(__name__)

# 创建一个队列
task_queue = queue.Queue()
task_id_list = []
# tmp文件夹
tmp_root_path = "./tmp"
os.makedirs(tmp_root_path, exist_ok=True)


@app.route('/api/buildVideo', methods=['POST'])
def build_video():
    # 验证POST请求的数据
    if 'driven_audio' not in request.files:
        return jsonify({'error': 'driven_audio can not be empty'}), 400
    if 'source_image' not in request.files:
        return jsonify({'error': 'source_image can not be empty'}), 400
    # 格式化POST数据
    args = build_args(request)
    # 提交进入队列
    task_queue.put(args)
    task_id = args.task_id
    task_id_list.append(task_id)
    print(task_id)
    # 返回处理结果
    return jsonify({'task_id': task_id})


@app.route('/api/getVideo', methods=['GET'])
def get_video():
    task_id = request.args.get('task_id')

    if task_id in task_id_list:
        # 返回 task_id 在 task_id_list 中的索引位置
        index = task_id_list.index(task_id)
        return jsonify({'index': index + 1, "total": len(task_id_list)})

    video_path = f'{tmp_root_path}/{task_id}/result.mp4'
    if os.path.isfile(video_path):
        # 使用 send_file 函数返回视频文件
        return send_file(video_path, as_attachment=True)

    if os.path.isdir(f'{tmp_root_path}/{task_id}'):
        # 抛出异常，文件夹存在但文件不存在的情况
        return jsonify({'error': task_id + " video build exception"})
    # 抛出异常, taskId不存在异常
    return jsonify({'error': task_id + " is not exist"})


def build_args(req):
    # 生成task_id(随机UUID)
    task_id = str(uuid.uuid4()).replace('-', '')
    driven_audio = req.files['driven_audio']
    source_image = req.files['source_image']
    # 创建保存文件的目录，并保存文件
    tmp_path = os.path.join(tmp_root_path, task_id)
    os.makedirs(tmp_path, exist_ok=True)
    driven_audio_path = os.path.join(tmp_path, "driven_audio" + os.path.splitext(driven_audio.filename)[1])
    source_image_path = os.path.join(tmp_path, "source_image" + os.path.splitext(source_image.filename)[1])
    driven_audio.save(driven_audio_path)
    source_image.save(source_image_path)
    # 组装参数
    args = {
        "task_id": task_id,
        "tmp_path": tmp_path,
        "driven_audio": driven_audio_path,
        "source_image": source_image_path,
    }
    args = build_common_args(args)
    # 转成object
    return types.SimpleNamespace(**args)


def build_common_args(args):
    args["ref_eyeblink"] = None
    args["ref_pose"] = None
    args["checkpoint_dir"] = './checkpoints'
    args["pose_style"] = int(0)
    args["batch_size"] = int(2)
    args["size"] = int(256)
    args["expression_scale"] = float(1.)
    args["input_yaw"] = None
    args["input_pitch"] = None
    args["input_roll"] = None
    # enhancer: default None [gfpgan, RestoreFormer]
    args["enhancer"] = "gfpgan"
    args["background_enhancer"] = None
    args["cpu"] = bool(False)
    args["face3dvis"] = bool(False)
    # still: default False
    args["still"] = bool(True)
    # preprocess: default crop ['crop', 'extcrop', 'resize', 'full', 'extfull']
    args["preprocess"] = 'full'
    args["verbose"] = bool(False)
    args["old_version"] = bool(False)

    # net structure and parameters
    args["net_recon"] = 'resnet50'
    args["init_path"] = None
    args["use_last_fc"] = bool(False)
    args["bfm_folder"] = './checkpoints/BFM_Fitting/'
    args["bfm_model"] = 'BFM_model_front.mat'

    # default renderer parameters
    args["focal"] = float(1015.)
    args["center"] = float(112.)
    args["camera_d"] = float(10.)
    args["z_near"] = float(5.)
    args["z_far"] = float(15.)

    # CUDA
    if torch.cuda.is_available() and not args["cpu"]:
        args["device"] = "cuda"
    else:
        args["device"] = "cpu"

    return args


def generate(args):
    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.tmp_path, "result")
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size,
                                args.old_version, args.preprocess)

    # init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)

    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, args.preprocess, \
                                                                           source_image_flag=True, pic_size=args.size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess,
                                                                  source_image_flag=False)
    else:
        ref_eyeblink_coeff_path = None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink:
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ = preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess,
                                                                  source_image_flag=False)
    else:
        ref_pose_coeff_path = None

    # audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    # 3dface render
    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))

    # coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                               batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                               expression_scale=args.expression_scale, still_mode=args.still,
                               preprocess=args.preprocess, size=args.size)

    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                         enhancer=args.enhancer, background_enhancer=args.background_enhancer,
                                         preprocess=args.preprocess, img_size=args.size)

    shutil.move(result, save_dir + '.mp4')
    print('The generated video is named:', save_dir + '.mp4')

    if not args.verbose:
        shutil.rmtree(save_dir)


# 创建视频生成任务start
def generate_video_task():
    while True:
        args = None
        try:
            # 从队列中获取数据
            args = task_queue.get(timeout=1)
            # 执行视频生成
            if args is not None:
                generate(args)
        except queue.Empty:
            pass
        except Exception as e:
            # 处理其他异常
            print("Exception occurred:", e)
        finally:
            if args is not None:
                task_id_list.remove(args.task_id)

        time.sleep(1)  # 等待1秒


# 创建并启动线程
thread = threading.Thread(target=generate_video_task)
thread.start()


# 创建视频生成任务end


# 定时删除超过24h文件夹start
def delete_old_folders_task():
    while True:
        try:
            current_time = time.time()
            with app.app_context():
                for folder_name in os.listdir(tmp_root_path):
                    folder_full_path = os.path.join(tmp_root_path, folder_name)
                    if os.path.isdir(folder_full_path) and current_time - os.path.getctime(
                            folder_full_path) > 1 * 60 * 60:
                        shutil.rmtree(folder_full_path)
                        print("delete the 24h older folder:", folder_full_path)
        except Exception as e:
            # 处理其他异常
            print("delete old folders exception occurred:", e)
        time.sleep(1 * 60 * 60)


delete_old_folders_thread = threading.Thread(target=delete_old_folders_task)
delete_old_folders_thread.start()


# 定时删除超过24h文件夹end

# 历史任务初始化start
def submit_task(task_id):
    # 将任务ID提交到队列中的代码
    task_tmp_path = os.path.join(tmp_root_path, task_id)
    args = {
        "task_id": task_id,
        "tmp_path": task_tmp_path,
        "driven_audio": get_driven_audio_path(task_tmp_path),
        "source_image": get_source_image_path(task_tmp_path),
    }
    args = build_common_args(args)
    # 转成object
    args_object = types.SimpleNamespace(**args)
    # 提交进入队列
    task_queue.put(args_object)
    task_id_list.append(task_id)
    print('Loading old task:', task_id)


def get_driven_audio_path(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if "driven_audio" in file_name:
                file_path = os.path.join(root, file_name)
                return file_path


def get_source_image_path(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if "source_image" in file_name:
                file_path = os.path.join(root, file_name)
                return file_path


def startup_submit_task():
    for folder_name in sorted(os.listdir(tmp_root_path),
                              key=lambda x: os.path.getctime(os.path.join(tmp_root_path, x))):
        folder_full_path = os.path.join(tmp_root_path, folder_name)
        if os.path.isdir(folder_full_path) and not os.path.exists(os.path.join(folder_full_path, "result.mp4")):
            submit_task(folder_name)


startup_submit_task()
# 历史任务初始化end


if __name__ == '__main__':
    app.run()
