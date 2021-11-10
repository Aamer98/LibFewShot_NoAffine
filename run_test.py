# -*- coding: utf-8 -*-
import argparse
from core import Trainer
from core import Test
from core.config import Config
import os
import sys

sys.dont_write_bytecode = True



parser = argparse.ArgumentParser(description='LibFewShot Training')

parser.add_argument('--data_root', default='./dataset/miniImageNet--ravi',  help='path to dataset')


Path_list = ['./results/Baseline-miniImageNet--ravi-Conv64F-5-1-Nov-08-2021-08-56-06',
             './results/Baseline-miniImageNet--ravi-Conv64F-5-5-Nov-08-2021-12-10-08',
             './results/MAML-miniImageNet--ravi-Conv64F-5-1-Nov-08-2021-08-56-07',
             './results/MAML-miniImageNet--ravi-Conv64F-5-5-Nov-08-2021-17-16-52',
             './results/ProtoNet-miniImageNet--ravi-Conv64F-5-1-Nov-08-2021-08-56-22',
             './results/ProtoNet-miniImageNet--ravi-Conv64F-5-20-Nov-08-2021-08-56-15',
             './results/ProtoNet-miniImageNet--ravi-Conv64F-5-5-Nov-08-2021-12-03-26']

args = parser.parse_args()
VAR_DICT = {
    "data_root": args.data_root,
    "test_epoch": 5,
    "device_ids": "0",
    "n_gpu": 1,
    "test_episode": 2000,
    "episode_size": 1,
    "test_way": 5}


if __name__ == "__main__":
    for PATH in Path_list:
        config = Config(os.path.join(test_in_bird-220100.out, "config.yaml"),
                    VAR_DICT).get_config_dict()
        test = Test(config, PATH)
        test.test_loop()


