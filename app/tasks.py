import os
import sys
import time
import queue
import types
import shutil
import traceback
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from app.args_utils import build_common_args

# 创建一个队列
task_queue = queue.Queue()
task_id_list = []
# tmp文件夹
tmp_root_path = "./tmp"


def get_tmp_root_path():
    if os.path.isabs(tmp_root_path):
        return tmp_root_path
    else:
        return os.path.join(os.path.dirname(__file__), "..", tmp_root_path)


def add_task(args):
    """
    添加任务队列
    """
    task_queue.put(args)
    task_id = args.task_id
    task_id_list.append(task_id)
    print("add task(" + task_id + ") to the queue")


def index_of_task_id_list(task_id):
    """
    获取task_id的顺序，及队列长度
    """
    if task_id in task_id_list:
        # 返回 task_id 在 task_id_list 中的索引位置
        return task_id_list.index(task_id), len(task_id_list)
    return -1, len(task_id_list)


def __generate(args):
    """
    生成视频
    """
    save_dir = os.path.join(args.tmp_path, "result")
    os.makedirs(save_dir, exist_ok=True)
    image_path = args.source_image
    audio_path = args.driven_audio
    checkpoint_dir = args.checkpoint_dir
    enhancer = args.enhancer  # Face enhancer, [gfpgan, RestoreFormer]
    still = args.still  # can crop back to the original videos for the full body aniamtion
    preprocess = args.preprocess  # ['crop', 'extcrop', 'resize', 'full', 'extfull'] [default 'crop']
    ref_eye_blink = args.ref_eyeblink  # path to reference video providing eye blinking
    ref_pose = args.ref_pose  # path to reference video providing pose
    pose_style = args.pose_style  # input pose style from [0, 46)  [default 0]
    device = args.device  # cuda / cpu
    input_yaw_list = args.input_yaw  # the input yaw degree of the user
    input_pitch_list = args.input_pitch  # the input pitch degree of the user
    input_roll_list = args.input_roll  # the input roll degree of the user
    img_size = args.size  # the image size of the face_render [default 256]
    batch_size = args.batch_size  # the batch size of face_render [default 2]
    expression_scale = args.expression_scale  # the batch size of face_render [default 1.]
    background_enhancer = args.background_enhancer  # background enhancer, [realesrgan] [default None]
    old_version = args.old_version  # use the pth other than safetensor version

    current_root_path = os.path.split(sys.argv[0])[0]

    # init config [wav2lip、audio2pose、audio2exp、freeView、BFM、mapping、face_render等]
    config_dir = os.path.join(current_root_path, 'src/config')
    sadtalker_paths = init_path(checkpoint_dir, config_dir, img_size, old_version, preprocess)

    # init model [预处理]
    preprocess_model = CropAndExtract(sadtalker_paths, device)
    # init model [音频->系数]
    audio_to_coefficient_model = Audio2Coeff(sadtalker_paths, device)
    # init model [系数->动画]
    animate_from_coefficient_model = AnimateFromCoeff(sadtalker_paths, device)

    # [预处理] 裁剪图像并从中提取3DMM
    # crop image and extract 3dmm from image
    print('3DMM Extraction for source image')
    image_frame_dir = os.path.join(save_dir, 'image_frame_dir')
    os.makedirs(image_frame_dir, exist_ok=True)
    image_coefficient_path, crop_pic_path, crop_info = preprocess_model.generate(image_path, image_frame_dir,
                                                                                 preprocess, True, img_size)
    if image_coefficient_path is None:
        print("Can't get the coefficient of the input source image")
        return

    # [预处理] 眨眼参考视频的3DMM提取
    if ref_eye_blink is not None:
        ref_eye_blink_video_name = os.path.splitext(os.path.split(ref_eye_blink)[-1])[0]
        ref_eye_blink_frame_dir = os.path.join(save_dir, ref_eye_blink_video_name)
        os.makedirs(ref_eye_blink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        eye_blink_coefficient_path, _, _ = preprocess_model.generate(ref_eye_blink, ref_eye_blink_frame_dir, preprocess,
                                                                     False)
    else:
        eye_blink_coefficient_path = None

    # [预处理] 姿势的参考视频的3DMM提取
    if ref_pose == ref_eye_blink:
        pose_coefficient_path = eye_blink_coefficient_path
    elif ref_pose is not None:
        ref_pose_video_name = os.path.splitext(os.path.split(ref_pose)[-1])[0]
        ref_pose_frame_dir = os.path.join(save_dir, ref_pose_video_name)
        os.makedirs(ref_pose_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing pose')
        pose_coefficient_path, _, _ = preprocess_model.generate(ref_pose, ref_pose_frame_dir, preprocess, False)
    else:
        pose_coefficient_path = None

    # [音频->系数]
    batch = get_data(image_coefficient_path, audio_path, device, eye_blink_coefficient_path, still)
    all_coefficient = audio_to_coefficient_model.generate(batch, save_dir, pose_style, pose_coefficient_path)

    # [3D Face Render]
    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        face3d_path = os.path.join(save_dir, '3dface.mp4')
        gen_composed_video(args, device, image_coefficient_path, all_coefficient, audio_path, face3d_path)

    # [系数->动画]
    face_render_data = get_facerender_data(all_coefficient, crop_pic_path, image_coefficient_path, audio_path,
                                           batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                           expression_scale, still, preprocess, img_size)
    result = animate_from_coefficient_model.generate(face_render_data, save_dir, image_path, crop_info,
                                                     enhancer, background_enhancer, preprocess, img_size)

    # 保存文件
    save_video_name = save_dir + '.mp4'
    shutil.move(result, save_video_name)
    print('The generated video is named:', save_video_name)


# 创建视频生成任务start
def generate_video_task():
    while True:
        args = None
        try:
            # 从队列中获取数据
            args = task_queue.get(timeout=1)
            # 执行视频生成
            if args is not None:
                print("start build " + args.task_id + " video")
                __generate(args)
        except queue.Empty:
            pass
        except BaseException as e:
            # 处理其他异常
            print("task(" + args.task_id + ") build video exception:", e)
            traceback.print_exc()
        finally:
            if args is not None:
                task_id_list.remove(args.task_id)

        time.sleep(1)  # 等待1秒


# 创建视频生成任务end


# 定时删除超过24h文件夹start
def delete_old_folders_task(app):
    while True:
        print("delete old folders task start...")
        try:
            current_time = time.time()
            with app.app_context():
                for folder_name in os.listdir(tmp_root_path):
                    folder_full_path = os.path.join(tmp_root_path, folder_name)
                    if os.path.isdir(folder_full_path) and current_time - os.path.getctime(
                            folder_full_path) > 24 * 60 * 60:
                        shutil.rmtree(folder_full_path)
                        print("delete the 24h older folder:", folder_full_path)
        except BaseException as e:
            # 处理其他异常
            print("delete old folders exception occurred:", e)
        print("delete old folders task end...")
        time.sleep(24 * 60 * 60)


# 定时删除超过24h文件夹end


# 历史任务初始化start
def submit_task(task_id):
    # 将任务ID提交到队列中的代码
    task_tmp_path = os.path.join(tmp_root_path, task_id)
    args = {
        "task_id": task_id,
        "tmp_path": task_tmp_path,
        "driven_audio": _get_driven_audio_path(task_tmp_path),
        "source_image": _get_source_image_path(task_tmp_path),
    }
    args = build_common_args(args)
    # 转成object
    args_object = types.SimpleNamespace(**args)
    # 提交进入队列
    add_task(args_object)


def _get_driven_audio_path(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if "driven_audio" in file_name:
                file_path = os.path.join(root, file_name)
                return file_path


def _get_source_image_path(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if "source_image" in file_name:
                file_path = os.path.join(root, file_name)
                return file_path


def loading_old_task():
    print("loading old task start...")
    for folder_name in sorted(os.listdir(tmp_root_path),
                              key=lambda x: os.path.getctime(os.path.join(tmp_root_path, x))):
        folder_full_path = os.path.join(tmp_root_path, folder_name)
        if os.path.isdir(folder_full_path) and not os.path.exists(os.path.join(folder_full_path, "result.mp4")):
            submit_task(folder_name)
    print("loading old task end...")

# 历史任务初始化end
