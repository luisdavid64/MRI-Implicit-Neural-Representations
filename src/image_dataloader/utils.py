def get_data_loader(data, data_path, img_dim, img_slice, train, batch_size, num_workers=4, data_idx=False):
    """

    :param data: str, kind of data 'brain' or 'knee'
    :param data_path: str, data path (or folder)
    :param img_dim: tuple, c x h x w
    :param img_slice: int, slice number
    :param train: True or False
    :param batch_size: int, bs
    :param num_workers: int, default 4
    :param data_idx: bool, return data index.
    :return: data loader
    """

