from Tea_MRBD import Model
from options.TrainTeacher_options import Options  # ready
from dataset.data_loader import TrainDataLoader, TestDataLoader
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    config = Options().param_setting()
    train_loader = TrainDataLoader(config.data_dir, config.batch_size, config.image_size,
                                   config.crop_size, config.num_workers)()

    val_loader = TrainDataLoader(config.val_dir, 9, config.image_size,
                                   config.crop_size, config.num_workers)()  # 和上述操作一样，只不过这里是验证集，并且batch_size=1

    model = Model(train_loader, val_loader, config)
    model.train()
