from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset


def get_tensor_loader(*datas, batch_size, shuffle=True, drop_last=False, **kwargs):
    """得到一个简单的DataLoader对象

    :param datas: tensor组成的 list 或 tuple

    :param batch_size: 批量大小

    :param shuffle: 是否打乱

    :param drop_last: 对最后不足batch_size大小的批次选择丢弃或保留
    """
    dset = TensorDataset(*datas)
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)