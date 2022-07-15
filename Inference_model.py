# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import argparse
from .configs.base_config import cfg, cfg_from_file, cfg_update
from .utils.misc import set_logger
from .runners.sparenet_runner import sparenetRunner
from .runners.grnet_runner import grnetRunner
from easydict import EasyDict

def get_args_from_command_line(additional_args):
    """
    config the parameter
    """
    parser = argparse.ArgumentParser(description="The argument parser of R2Net runner")

    # choose model
    parser.add_argument("--model", type=str, default=additional_args.get("model", "sparenet"), help="sparenet, atlasnet, msn, grnet")

    # choose test mode
    parser.add_argument("--test_mode", default=additional_args.get("test_mode", "default"), help="default, vis, render, kitti", type=str)

    # choose load model
    parser.add_argument("--weights", dest="weights", help="Initialize network from the weights file", default=additional_args.get("ckpt", None))

    # setup gpu
    parser.add_argument("--gpu", dest="gpu_id", help="GPU device to use", default="0", type=str)

    # setup workdir
    parser.add_argument("--output", help="where to save files", default=additional_args.get("outputs", "/home/halperin/ML3D/Outputs/Completion"), type=str)

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
        self._model, self._cfg  = self.make_model(args)
        
    
    def make_model(self, args=dict()):
        # args = get_args_from_command_line(args)
        
        cfg_from_file(args.RECONSTRUCTION.local_dir + "/configs/" + args.RECONSTRUCTION.model + ".yaml")
        if args.RECONSTRUCTION.test_mode is not None:
            cfg.TEST.mode = args.RECONSTRUCTION.test_mode
        output_dir = cfg_update(args)

        # Add project arguments to cfg
        cfg["PROJECT"] = EasyDict()
        #for k, v in args.items():
        cfg["PROJECT"].update(args["RECONSTRUCTION"])
        #    break
        # model.test()

        
        if cfg.PROJECT.model == "sparenet":
            model = sparenetRunner(cfg, logger=None)
        elif cfg.PROJECT.model == "grnet":
            model = grnetRunner(cfg, logger=None)
        else:
            raise Exception("--model is not a valid model name: {}".format(cfg.PROJECT.model))

        return model, cfg

    @property
    def model(self):
        return self._model
    
    @property
    def cfg(self):
        return self._cfg

    def test(self):
        self.model.test()
    
