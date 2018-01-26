import os

from google.protobuf import text_format

from protos import config_pb2


def get_configs_from_file(config_file):
    """
    get config file json config file
    :param config_file: config file path
    """
    if not os.path.exists(config_file):
        raise ValueError('%s do not exist' % config_file)

    config_info = config_pb2.TrainConfig()
    with open(config_file, 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config_info)

    return config_info

def set_config_to_file(config, file_path):
    return


def set_default_config_to_file(file_path):
    return


if __name__ == '__main__':
    # cfg = config_pb2.TrainConfig()
    # cfg = get_configs_from_file("../config/train.config")
    # print('cfg.train_data_root = %s' % cfg.train)
    config_info = config_pb2.TrainConfig()
    print(config_info.__str__())

