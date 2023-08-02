import os, uuid, types, torch

import config


def build_args(req):
    """
    解析转换请求参数
    """
    # 生成task_id(随机UUID)
    task_id = str(uuid.uuid4()).replace('-', '')
    driven_audio = req.files['driven_audio']
    source_image = req.files['source_image']
    # 创建保存文件的目录，并保存文件
    tmp_path = os.path.join(config.TMP_PATH, task_id)
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
    """
    添加公共参数，给默认值
    """
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
