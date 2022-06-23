# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import argparse
from .configs.base_config import cfg, cfg_from_file, cfg_update
from .utils.misc import set_logger
from .runners.sparenet_runner import sparenetRunner
from easydict import EasyDict

def get_args_from_command_line(additional_args):
    """
    config the parameter
    """
    parser = argparse.ArgumentParser(description="The argument parser of R2Net runner")

    # choose model
    parser.add_argument("--model", type=str, default="sparenet", help="sparenet, atlasnet, msn, grnet")

    # choose test mode
    parser.add_argument("--test_mode", default=additional_args.get("test_mode", "default"), help="default, vis, render, kitti", type=str)

    # choose load model
    parser.add_argument("--weights", dest="weights", help="Initialize network from the weights file", default=additional_args.get("ckpt", None))

    # setup gpu
    parser.add_argument("--gpu", dest="gpu_id", help="GPU device to use", default="0", type=str)

    # setup workdir
    parser.add_argument("--output", help="where to save files", default=additional_args.get("output", "/home/halperin/ML3D/Outputs/Completion"), type=str)

    # choose train mode
    parser.add_argument("--gan", dest="gan", help="use gan", action="store_true", default=False)
    parser.add_argument("--local_dir", type=str, default="/home/halperin/ML3D/SpareNet", help="sparenet, atlasnet, msn, grnet")
    
    args = vars(parser.parse_args())
    for k, v in additional_args.items():
        if k not in args:
            parser.add_argument('--' + k, default=v)
    return parser.parse_args()



class SpareNet():
    
    def __init__(self, args) -> None:
        self.model = self.make_model(args)

    
    def make_model(self, args=dict()):
        args = get_args_from_command_line(args)

        # Set GPU to use
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        # update config
        
        cfg_from_file(args.local_dir + "/configs/" + args.model + ".yaml")
        if args.test_mode is not None:
            cfg.TEST.mode = args.test_mode
        output_dir = cfg_update(args)

        # Set up folders for logs and checkpoints
        # if not os.path.exists(cfg.DIR.logs):
        #     os.makedirs(cfg.DIR.logs)

        # logger = set_logger(os.path.join(cfg.DIR.logs, "log.txt"))
        # logger.info("save into dir: %s" % cfg.DIR.logs)

        # Add project arguments to cfg
        cfg["PROJECT"] = EasyDict()
        for k, v in vars(args).items():
            cfg["PROJECT"][k] = v
        # model.test()
        model = sparenetRunner(cfg, logger=None)

        return model

    @property
    def get_model(self):
        return self.model

    def test(self):
        self.model.test()
    
