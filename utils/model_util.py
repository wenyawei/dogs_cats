# coding:utf8
import os
from shutil import copyfile

import logging
import torch


def load(path):
    """
    load a model from checkpoint path
    :param checkpoint:
    :param path:
    :return:
    """

    if os.path.exists(path):
        checkpoint = torch.load(path)
    else:
        logging.exception('no checkpoint found at {path}'.format(path=path))
        raise("error")

    return checkpoint


def save(state, checkpoint_dir, is_best=False, name=None):
    """
    save the model as checkpoint
    :param state:
    :param checkpoint_dir:
    :param is_best:
    :param name:
    :return:
    """

    if name is None:
        prefix = os.path.join(checkpoint_dir, state['model'] + '_')
        name = prefix + "%03d" % state['epoch'] + ".pth"
        # name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')

    torch.save(state, name)
    if is_best:
        copyfile(name, os.path.join(checkpoint_dir, 'model_best.pth'))
