import os
import logging
import argparse

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision
from torchnet import meter
from tqdm import tqdm

from data.dataset import DogCat
from utils import config_util
from utils import visualizer
from utils import model_util

parser = argparse.ArgumentParser(description='Dog And Cat model')
parser.add_argument('--config-file', default='./config/train.config', type=str, metavar='PATH',
                    help='path to config file')

args = parser.parse_args()
print("config-file %s" % args.config_file)

config = config_util.get_configs_from_file(args.config_file)


def train():
    """
    train function
    :return:
    """

    vis = visualizer.Visualizer(config.visdom_env)

    # step1: configure model
    model = torchvision.models.densenet121(pretrained=False, num_classes=2)
    if config.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # step2: data
    train_data = DogCat(config.train_data_root, train=True)
    val_data = DogCat(config.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, config.batch_size,
                                  shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_data, config.batch_size,
                                shuffle=False, num_workers=config.num_workers)

    # step3: criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    # step4: meters
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # train
    model.train()
    best_acc = -1.0
    start_epoch = -1

    # optionally resume from a checkpoint
    state = dict()
    if config.load_model_path:
        logging.info('Loading checkpoint from {path}'.format(path=config.load_model_path))
        state = model_util.load(config.load_model_path)
        start_epoch = state['epoch']
        best_acc = state['accuracy']
        model.load_state_dict(state['state_dic'])
        optimizer.load_state_dict(state['optimizer'])
        logging.info('Loaded checkpoint from {path}'.format(path=config.load_model_path))

    for epoch in range(start_epoch + 1, config.max_epoch):

        logging.info('epoch = %d' % epoch)

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_dataloader), total=len(train_data)):

            # train model
            input_var = Variable(data)
            target_var = Variable(label)
            if config.use_gpu:
                input_var = input_var.cuda()
                target_var = target_var.cuda()

            optimizer.zero_grad()
            score = model(input_var)
            loss = criterion(score, target_var)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.data[0])
            confusion_matrix.add(score.data, target_var.data)

            if ii % config.print_freq == config.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])

                # 进入debug模式
                # if os.path.exists(config.debug_file):
                #     import ipdb;
                #     ipdb.set_trace()

        # validate and visualize
        val_cm, val_accuracy = val(model, val_dataloader)

        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))

        is_best = val_accuracy > best_acc
        best_acc = max(val_accuracy, best_acc)

        logging.info("epoch:{epoch},lr:{lr},loss:{loss},acc:{acc} train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], acc=val_accuracy, val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))

        state['epoch'] = epoch
        state['model'] = config.model
        state['state_dic'] = model.state_dict()
        state['accuracy'] = val_accuracy
        state['optimizer'] = optimizer.state_dict()
        model_util.save(state, config.checkpoint_dir, is_best)

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * config.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


def eval():
    """
    evaluation
    :return:
    """
    return


def inference():
    """
    inference
    :return:
    """
    logging.debug('Begin to inference')

    # step1: configure model
    model = torchvision.models.densenet121(pretrained=False, num_classes=2)
    if config.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    logging.info('Loading checkpoint from {path}'.format(path=config.load_model_path))
    state = model_util.load(config.load_model_path)
    model.load_state_dict(state['state_dic'])
    model.eval()
    logging.info('Loaded checkpoint from {path}'.format(path=config.load_model_path))

    # dataset
    test_data = DogCat(config.test_data_root, test=True)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    results = []
    for ii, (data, path) in enumerate(test_dataloader):
        input_var = torch.autograd.Variable(data, volatile=True)
        if config.use_gpu:
            input_var = input_var.cuda()
        score = model(input_var)
        probability = torch.nn.functional.softmax(score)[:, 0].data.tolist()
        # label = score.max(dim = 1)[1].data.tolist()

        batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]

        results += batch_results
    write_csv(results, config.result_file)

    logging.debug('End to inference')

    return results


def val(model, dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label.type(torch.LongTensor), volatile=True)
        if config.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.type(torch.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def write_csv(results, file_name):
    """

    :param results:
    :param file_name:
    :return:
    """
    # import numpy as np
    # results_np = np.array(results)
    # results_np[results_np[:, 0].argsort()]

    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def main():
    """
    main function
    :return:
    """
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

    print('log_file = {log_file}'.format(log_file=config.log_file))

    logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT,
                        level=logging.DEBUG, filename=config.log_file)

    if config.mode == 1:
        train()
    elif config.mode == 2:
        inference()
    elif config.mode == 3:
        eval()
    else:
        raise ("config mode error %d" % config.mode)


if __name__ == '__main__':
    main()
