from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset


class ChannelFistDataset(data.Dataset):
    def __init__(self, *datas):
        """
        datas[0] 的形状需要是 (num_channels, num_data, ....)

        data[other] 的形状需要时 (num_data, ...)
        """
        super(ChannelFistDataset, self).__init__()
        for d in datas:
            print(d.shape)
        self.datas = datas

    def __getitem__(self, index):
        return [dt[index] if i > 0 else dt[:, index] for i, dt in enumerate(self.datas)]

    def __len__(self):
        return self.datas[0].shape[1]


def get_channel_first_loader(*datas, batch_size, shuffle=True, drop_last=False, **kwargs):
    dset = ChannelFistDataset(*datas)
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)


def get_tensor_loader(*datas, batch_size, shuffle=True, drop_last=False, **kwargs):
    """得到一个简单的DataLoader对象

    :param datas: tensor组成的 list 或 tuple

    :param batch_size: 批量大小

    :param shuffle: 是否打乱

    :param drop_last: 对最后不足batch_size大小的批次选择丢弃或保留
    """
    dset = TensorDataset(*datas)
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)