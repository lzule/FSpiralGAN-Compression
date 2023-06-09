# -*-coding:utf-8-*-
import os
import argparse
import torch

parser = argparse.ArgumentParser()

config = parser.parse_args()


class Options(object):
    def __init__(self):
        self.root_path = '/home/lizl/snap/third-stage/Distill-dwt/trained_model'
        self.parser = argparse.ArgumentParser()

    def param_setting(self):  #Li 引入参数
#------------------------------------other_relate------------------------------------
        self.parser.add_argument('--num_workers', type=int, default=4) 
        self.parser.add_argument('--out_dir', type=str,
                                 default='dis_result_train/',
                                 help="data folder")

#------------------------------------train_data_relate------------------------------------
        self.parser.add_argument('--data_dir', type=str, default='/home/lizl/snap/second-stage/data/',
                                 help="data folder")
        self.parser.add_argument('--val_dir', type=str, default='/home/lizl/snap/second-stage/data/val',
                                 help="data folder")
        
#------------------------------------model_train_relate------------------------------------
        self.parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
        self.parser.add_argument('--teacher_global_G_ngf', type=int, default=16,
                                 help='number of teacher conv filters in the first layer of G')
        self.parser.add_argument('--student_global_G_ngf', type=int, default=8,
                                 help='number of student conv filters in the first layer of G')
        self.parser.add_argument('--global_D_ndf', type=int, default=32,  # ready
                                 help='number of conv filters in the first layer of MBD')

#------------------------------------model_optimizer_relate------------------------------------
        self.parser.add_argument("--global_glr", type=float, default=0.0001,  #Li
                                 help="initial learning rate for adam")
        self.parser.add_argument("--global_dlr", type=float, default=0.0001,  #Li 和上面这两个参数的区别在哪里？
                                 help="initial learning rate for adam")
        self.parser.add_argument('--beta1', type=float, default=0.5,
                                 help='beta1 for Adam optimizer')
        self.parser.add_argument('--beta2', type=float, default=0.999,
                                 help='beta2 for Adam optimizer')
        
#------------------------------------gpu_train_relate------------------------------------
        self.parser.add_argument('--gpu_id', type=int, default=1,
                                 help='gpu ids: e.g.0,1,2 use -1 for CPU')

#------------------------------------resume_relate------------------------------------
        self.parser.add_argument('--Train_resume', type=int, default=None,
                                 help='test model from this step')
        self.parser.add_argument('--Super_student', type=int, default=None,
                                 help='test model from this step')
        self.parser.add_argument('--Super_teacher', type=int, default=None,
                                 help='test model from this step')
        self.parser.add_argument('--Super_discri', type=int, default=None,
                                 help='test model from this step')
        self.parser.add_argument('--student_dir', type=str, default=os.path.join(self.root_path, 'Student'),
                                 help="teacher_model data folder")
        self.parser.add_argument('--teacher_dir', type=str, default=os.path.join(self.root_path, 'Teacher'),
                                 help="teacher_model data folder")
        self.parser.add_argument('--Dmodel_dir', type=str, default=os.path.join(self.root_path, 'Discr'),
                                 help="teacher_model data folder")
        
#------------------------------------train_relate------------------------------------
        self.parser.add_argument('--num_epoch', type=int, default=100,
                                 help='number of total iterations for training net')
        self.parser.add_argument('--test_epoch', type=int, default=0,
                                 help='test model from this step')
        self.parser.add_argument('--num_recurrent', type=int, default=10,
                                 help='number of recurrent')
        self.parser.add_argument('--num_branch', type=int, default=1,
                                 help='number of branch in each MBDiscriminator')
        
#------------------------------------save_dir_relate------------------------------------
        self.parser.add_argument('--sample_dir', type=str,
                                 default=os.path.join(self.parser.get_default("out_dir"), 'samples'))
        self.parser.add_argument('--metric_dir', type=str,
                                 default=os.path.join(self.parser.get_default("out_dir"), 'metric_dir'))
        self.parser.add_argument('--model_save_dir', type=str,
                                 default=os.path.join(self.parser.get_default("out_dir"), 'models'))
        self.parser.add_argument('--result_dir', type=str,
                                 default=os.path.join(self.parser.get_default("out_dir"), 'results'))

#------------------------------------predispose_relater------------------------------------
        self.parser.add_argument('--image_size', type=int, default=256,
                                 help='image resolution')
        self.parser.add_argument('--crop_size', type=int, default=256,
                                 help='crop size for the RaFD dataset')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='mini-batch size')

#------------------------------------Lr_decay_realte------------------------------------
        self.parser.add_argument('--lr_update_step', type=int, default=50)  #Li 这个参数的意思？
        self.parser.add_argument('--num_epoch_decay', type=int, default=96,
                                 help='number of iterations for decaying lr')

#------------------------------------log_realte------------------------------------
        self.parser.add_argument('--log_step', type=int, default=1)
        self.parser.add_argument('--log_dir', type=str,
                                 default=os.path.join(self.parser.get_default("out_dir"), 'logs'))
        self.parser.add_argument('--use_tensorboard', type=self.str2bool, default=True,
                                 help='record or not')

#------------------------------------Dis_train_relater------------------------------------
        self.parser.add_argument('--dis_teacher_epoch', type=int, default=100,
                                 help='number of total iterations for training net')
        self.parser.add_argument('--lambda_dwt', type=float, default=50,  # ready
                                 help='the super value of lambda_dwt')
        
#------------------------------------Super_train_relater------------------------------------
        self.parser.add_argument('--mapping_layers', type=list, default=['con3',
                                                                         'decon6'],
                                 help='target of distill map layers')
        self.parser.add_argument('--n_channels', type=dict, default={'con1':[16, 12, 8], 'con2':[16, 12, 8], 'con3':[16, 12, 8], 
                                                                     'con4':[16, 12, 8],'con5':[16, 12, 8],
                                                                     'RCAB1':[[16, 12, 8], [16, 12, 8]], 'decon4':[[16, 12, 8], [16, 12, 8]],
                                                                     'RCAB2':[[32, 24, 16, 12, 8], [32, 24, 16, 12, 8]], 'decon5':[[16, 12, 8], [16, 12, 8]],
                                                                     'RCAB3':[[32, 24, 16, 12, 8], [32, 24, 16, 12, 8]], 'decon6':[[16, 12, 8], [16, 12, 8]],
                                                                     'RCAB4':[[32, 24, 16, 12, 8], [32, 24, 16, 12, 8]], 'decon7':[[16, 12, 8], [16, 12, 8]],
                                                                     'decon8':[[16, 12, 8], [16, 12, 8]]}, help='the Super net')
        
#------------------------------------Loss_lambda_relate------------------------------------
        self.parser.add_argument('--lambda_global_gan', type=float, default=1,
                                 help='beta2 for Adam optimizer')
        self.parser.add_argument('--lambda_global_l1', type=float, default=100,
                                 help='beta2 for Adam optimizer')
        self.parser.add_argument('--lambda_angular', type=float, default=0.1,
                                 help='beta2 for Adam optimizer')
        self.parser.add_argument('--lambda_distill', type=float, default=1,  # ready
                                 help='the super value of lambda_distill')
        
        config = self.parser.parse_args()
        self.param_init(config)
        print(config)
        return config

    def str2bool(self, v):
        return v.lower() in ('true')

    def param_init(self, config):
        # Create directories if not exist.
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
        if not os.path.exists(config.model_save_dir):
            os.makedirs(config.model_save_dir)
        if not os.path.exists(config.sample_dir):
            os.makedirs(config.sample_dir)
        if not os.path.exists(config.result_dir):
            os.makedirs(config.result_dir)

        if config.mode == "test":
            config.batch_size = 1

        gpu_id = int(config.gpu_id)
        torch.cuda.set_device(gpu_id)


if __name__ == '__main__':
    config = Options().param_setting()
