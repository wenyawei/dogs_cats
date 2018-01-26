from utils import config_util
from protos import config_pb2
from google.protobuf import text_format


def main():
    config_info = config_pb2.TrainConfig()
    config_info.lr = 0.1
    res = [f.name for f in config_info.DESCRIPTOR.fields]
    print(res)
    print(config_info.ListFields())

    print(config_info.__str__())
    string_info = text_format.MessageToString(config_info)
    print(string_info)


if __name__ == '__main__':
    main()
