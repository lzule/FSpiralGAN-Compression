# -*-coding:utf-8-*-
from torch.nn import init
import torch
import torch.nn.functional as F
from utils.objective import GANLoss, MSELoss, AngularLoss
from torch.nn.parallel import gather, parallel_apply, replicate
from torch import nn
# from nets.Stu import Student_G
from nets.Teacher_Generator import Teacher_G as Student_G
import time
from torchvision.utils import save_image
import os
import datetime
from nets.Discriminator import Discriminator
from torch.utils.tensorboard import SummaryWriter
import random
from utils.UCIQE_utils import getUCIQE
import numpy as np
from utils.UIQM_utils import getUIQM


class Model(object):
    def __init__(self, data_loader, val_loader, config):
#-----------------------dataloader_relate------------------------------
        self.train_loader = data_loader
        self.val_loader = val_loader

#------------------------model_train_relate----------------------------
        self.mode = config.mode
        self.teacher_global_g_dim = config.teacher_global_G_ngf  # ready
        self.student_global_g_dim = config.student_global_G_ngf  # ready
        self.global_d_dim = config.global_D_ndf

#-----------------------model_optimizer_relate--------------------------
        self.global_glr = config.global_glr
        self.global_dlr = config.global_dlr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

#---------------------------gpu_train_relate-----------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_id = [0]

#----------------------------resume_relate-------------------------------
        self.Train_resume = config.Train_resume
        self.Super_train_student_epoch = config.Super_student
        self.Super_train_teacher_epoch = config.Super_teacher
        self.Super_train_discri_epoch = config.Super_discri
        self.student_dir = config.student_dir
        self.teacher_dir = config.teacher_dir
        self.Dmodel_dir = config.Dmodel_dir

#----------------------------train_relate--------------------------------
        self.num_epoch = config.num_epoch
        self.test_epoch = config.test_epoch
        self.num_recurrent = config.num_recurrent
        self.num_branch = config.num_branch

#----------------------------save_dir_relate----------------------------
        self.sample_dir = config.sample_dir
        self.metric_dir = config.metric_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        
#----------------------------Lr_decay_realte----------------------------
        self.lr_update_step = config.lr_update_step
        self.num_epoch_decay = config.num_epoch_decay
        
#---------------------------log_realte-----------------------------------
        self.log_step = config.log_step
        self.log_dir = config.log_dir
        self.use_tensorboard = config.use_tensorboard
        self.writer = SummaryWriter(self.log_dir) if self.use_tensorboard else None
        self.global_loss = dict()
        self.print_log = open("./Teacher_train_log.txt", 'w')

#----------------------------Super_train_relater---------------------------
        self.mapping_layers = config.mapping_layers  # ready
        self.n_channels = config.n_channels
        self.configs = dict()

#----------------------------Loss_lambda_relate---------------------------
        self.lambda_global_gan = config.lambda_global_gan
        self.lambda_global_l1 = config.lambda_global_l1
        self.lambda_angular = config.lambda_angular
        self.lambda_distill = config.lambda_distill
        if self.mode.lower() == "train":
            self.gan_loss = GANLoss().to(self.device)
            self.global_l1_loss = MSELoss().to(self.device)  # L2损失函数
            self.angular_loss = AngularLoss().to(self.device)  # 角度损失函数
        self.build_model()

    # ?不太理解
    def restore_Super_model(self):
        # teacher_model
        self.my_print('Loading the Super models ...')
        teacher_G_path = os.path.join(self.teacher_dir, '{}-global_G.ckpt'.format(self.Super_train_teacher_epoch))
        self.teacher_G.load_state_dict(
            torch.load(teacher_G_path, map_location=lambda storage, loc: storage))
        # discri_model
        for i in range(self.num_recurrent):  # i=[0:10]
            for j in range(self.num_branch):  # j=[0:1]
                self.__getattribute__("D_" + str(i + 1) + "_" + str(j + 1)).load_state_dict(
                    torch.load(os.path.join(self.Dmodel_dir,
                                            ("{}-D_" + str(i + 1) + "_" + str(j + 1) + ".ckpt").format(self.Super_train_discri_epoch)),
                               map_location=lambda storage, loc: storage))
        # student_model
        student_G_path = os.path.join(self.student_dir, '{}-global_G.ckpt'.format(self.Super_train_student_epoch))
        self.student_G.load_state_dict(
            torch.load(student_G_path, map_location=lambda storage, loc: storage))
    
    def restore_Norm_model(self, resume_epoch):
        self.my_print('Resuming the trained models ...')
        teacher_G_path = os.path.join(self.teacher_dir, '{}-global_G.ckpt'.format(self.Super_train_teacher_epoch))
        self.teacher_G.load_state_dict(
            torch.load(teacher_G_path, map_location=lambda storage, loc: storage))
        student_G_path = os.path.join(self.model_save_dir, '{}-global_G.ckpt'.format(resume_epoch))
        self.student_G.load_state_dict(
            torch.load(student_G_path, map_location=lambda storage, loc: storage))  # 加载生成模型‘100-global_G.ckpt’
        for i in range(self.num_recurrent):  # i=[0:10]
            for j in range(self.num_branch):  # j=[0:1]
                self.__getattribute__("D_" + str(i + 1) + "_" + str(j + 1)).load_state_dict(
                    torch.load(os.path.join(self.model_save_dir,
                                            ("{}-D_" + str(i + 1) + "_" + str(j + 1) + ".ckpt").format(resume_epoch)),
                            map_location=lambda storage, loc: storage))
        for net, name in zip(self.netAs, self.mapping_layers):
            net.weight.data = torch.load(
                os.path.join(self.model_save_dir, str(resume_epoch) + '-' + name + '_weight.pt'))
            net.bias.data = torch.load(os.path.join(self.model_save_dir, str(resume_epoch) + '-' + name + '_bias.pt'))

    def my_print(self, text):
        print(text)
        print(text, file=self.print_log)

    def build_model(self):
        """create a generator and a discriminators"""
        self.student_G = Student_G(self.student_global_g_dim)
        self.print_network(self.student_G, 'student_net')
        # 在类实例的每个属性进行赋值时，都会首先调用__setattr__()方法，并在__setattr__()方法中将属性名和属性值添加到类实例的__dict__属性中。
        for i in range(self.num_recurrent):  # 组成鉴别器， 一共含有num_recurrent个相同的鉴别器
            for j in range(self.num_branch):
                self.__setattr__("D_" + str(i + 1) + "_" + str(j + 1),
                                 # setattr(object,name,value) object--对象,name--字符串，对象属性,value--属性值
                                 Discriminator(self.global_d_dim))  # Lh "D_"+str(i+1)+"_"+str(j+1)为鉴别器
                self.__setattr__("d" + str(i + 1) + "_" + str(j + 1) + "_optimizer",
                                 torch.optim.Adam(
                                     self.__getattribute__("D_" + str(i + 1) + "_" + str(j + 1)).parameters(),
                                     lr=self.global_dlr, betas=(self.beta1,
                                                                self.beta2)))  # Lh "d"+str(i+1)+"_"+str(j+1)+"_optimizer"为优化器adam，用来更新鉴别器的参数paramenters
                self.print_network(self.__getattribute__("D_" + str(i + 1) + "_" + str(j + 1)),
                                   "D_" + str(i + 1) + "_" + str(j + 1))
                self.__getattribute__("D_" + str(i + 1) + "_" + str(j + 1)).to(self.device)
        self.student_G_optimizer = torch.optim.Adam(self.student_G.parameters(), lr=self.global_glr,
                                                    betas=(self.beta1, self.beta2))
        self.student_G.to(self.device)  # ready 将网络送进GPU

    def reset_grad(self):  # Ready  教师网络就不需要重置梯度了、学生网络和鉴别器仍然需要置梯度
        """Reset the gradient buffers."""
        self.student_G_optimizer.zero_grad()  # 这一步是将生成器的参数梯度置零
        for i in range(self.num_recurrent):  # 这一步是将所有的鉴别器梯度置零
            for j in range(self.num_branch):
                self.__getattribute__("d" + str(i + 1) + "_" + str(j + 1) + "_optimizer").zero_grad()

    def print_network(self, model, name):
        """print the information of networks"""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        self.my_print(model)
        self.my_print(name)
        self.my_print("The number of parameters: {}".format(num_params))

    def update_lr(self, global_glr, global_dlr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.student_G_optimizer.param_groups:
            param_group['lr'] = global_glr

        for i in range(self.num_recurrent):
            for j in range(self.num_branch):
                for param_group in self.__getattribute__(
                        "d" + str(i + 1) + "_" + str(j + 1) + "_optimizer").param_groups:
                    param_group['lr'] = global_dlr

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def forward(self):
        # with torch.no_grad():
        #     _ = self.teacher_G(self.distored_real)
        self.student_G.configs = self.configs
        self.clean_fake_s = self.student_G(self.distored_real)
        self.distored_step2 = self.clean_fake_s.detach()  # 这一步是用来为下一次recurrent的输入做准备 # 在需要使用A网络的Variable进行B网络中的backprop操作，但又不想更新A网络梯度时，可以使用detach操作
        self.real_dc = torch.cat((self.distored_real, self.clean_real), dim=1)
        self.fake_dc_s = torch.cat((self.distored_real, self.clean_fake_s), dim=1)

    def backward_D(self, k):
        d_loss = 0.
        for l in range(self.num_branch):  # 这个模型有recurrent个判别器和一个生成器，每recurrent一次，就更新一次对应的判别器和全局的生成器
            global_d_out_real = self.__getattribute__("D_" + str(k + 1) + "_" + str(l + 1))(self.real_dc)
            global_d_loss_real = self.lambda_global_gan * self.gan_loss(global_d_out_real, True)
            global_d_out_fake_s = self.__getattribute__("D_" + str(k + 1) + "_" + str(l + 1))(
                self.fake_dc_s.detach())
            global_d_loss_fake_s = self.lambda_global_gan * self.gan_loss(global_d_out_fake_s, False)
            global_d_loss = (global_d_loss_real + global_d_loss_fake_s) * 0.5
            global_d_loss.backward()
            self.__getattribute__("d" + str(k + 1) + "_" + str(l + 1) + "_optimizer").step()
            d_loss += global_d_loss
        self.global_loss["Cycle" + str(k + 1) + "/d_loss"] = d_loss.item()

    def backward_G(self, k):
        g_loss_gan = 0.
        for l in range(self.num_branch):
            d_out = self.__getattribute__("D_" + str(k + 1) + "_" + str(l + 1))(self.fake_dc_s)
            g_loss_gan += self.lambda_global_gan * self.gan_loss(d_out, True) * 0.5

        global_loss_l1 = self.lambda_global_l1 * self.global_l1_loss(self.clean_real, self.clean_fake_s)
        angular_loss = self.lambda_angular * self.angular_loss(self.clean_real, self.clean_fake_s)
        g_loss = g_loss_gan + global_loss_l1 + angular_loss
        g_loss.backward()
        self.student_G_optimizer.step()
        self.global_loss["Cycle" + str(k + 1) + "/g_loss"] = g_loss_gan.item()
        self.global_loss["Cycle" + str(k + 1) + "/mse_loss"] = global_loss_l1.item()
        self.global_loss["Cycle" + str(k + 1) + "/ang_loss"] = angular_loss.item()

    def SaveLog(self, bz_iter, epoch):
        if (bz_iter + 1) % self.log_step == 0:
            et = time.time() - self.start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            global_str = ""
            for tag, value in self.global_loss.items():
                global_str += "{}: {:.6f}     ".format(tag, value)
            self.my_print("Elapsed [{}], Epoch [{}/{}], Iteration [{}/{}]:"
                          .format(et, epoch, self.num_epoch, bz_iter + 1, len(self.train_loader)))
            self.my_print(global_str)
            if self.use_tensorboard:
                for tag, value in self.global_loss.items():
                    self.writer.add_scalar(tag, value, len(self.train_loader) * epoch + bz_iter + 1)

    def TrainImage(self, epoch):
        with torch.no_grad():
            distored_fixed = self.muddy.to(self.device)
            clean_fixed = self.clean.to(self.device)
            clean_fake1 = self.student_G(distored_fixed)
            # clean_fake2 = self.teacher_G(distored_fixed)
            result_list = [distored_fixed, clean_fake1, clean_fixed]
            result_concat = torch.cat(result_list, dim=3)
            sample_path = os.path.join(self.sample_dir, '{}-epoch-images.jpg'.format(epoch))
            save_image((self.denorm(result_concat.data.cpu())), sample_path, nrow=2, padding=0)
            self.my_print('Saved real and fake images into {}...'.format(sample_path))

    def TestImage(self, epoch):
        if epoch >= self.test_epoch and epoch % 5 == 0:
            with torch.no_grad():
                self.my_print('Saving the val_dir real and fake images ...')
                for i, tensor_dic in enumerate(self.val_loader):
                    val_img, clean = tensor_dic["muddy"], tensor_dic["clean"]
                # for i, (val_img, _) in enumerate(self.val_loader):
                    second_muddy = val_img.to(self.device)
                    clean_fake_student = self.student_G(second_muddy)
                    # clean_fake_teacher = self.teacher_G(second_muddy)
                    result_path = os.path.join(self.result_dir, '{}-epoch-images'.format(epoch))
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)
                    second_muddy = second_muddy.data.cpu()
                    clean_fake_student = clean_fake_student.data.cpu()
                    # clean_fake_teacher = clean_fake_teacher.data.cpu()
                    compare_img_list = [second_muddy, clean_fake_student, clean.data.cpu()]
                    compare_img_concat = torch.cat(compare_img_list, dim=3)
                    save_image((self.denorm(compare_img_concat.data.cpu())),
                               os.path.join(result_path, str(i) + '_.jpg'), nrow=3, padding=0)
                self.my_print(
                    'Saved {} number of the val_dir real and fake images into {}...'.format(i, result_path))
        # with torch.no_grad():
        #     self.my_print('Saving the val_dir real and fake images ...')
        #     for i, (val_img, _) in enumerate(self.val_loader):
        #         second_muddy = val_img.to(self.device)
        #         clean_fake_student = self.student_G(second_muddy)
        #         metric_path = os.path.join(self.metric_dir, '{}-epoch-images'.format(epoch))
        #         if not os.path.exists(metric_path):
        #             os.makedirs(metric_path)
        #         clean_fake_student = clean_fake_student.data.cpu()
        #         save_image((self.denorm(clean_fake_student)),
        #                    os.path.join(metric_path, str(i) + '_.jpg'), nrow=1, padding=0)
        #     self.my_print(
        #         'Saved {} number of the val_dir real and fake images into {}...'.format(i, metric_path))

    def MetricImage(self, epoch):
        metric_path = os.path.join(self.metric_dir, '{}-epoch-images'.format(epoch))
        image_list = os.listdir(metric_path)
        UCIQE = list()
        for name in image_list:
            UCIQE.append(getUCIQE(os.path.join(metric_path, name)))
        # UIQM = getUIQM(metric_path)
        self.UCIQE = np.mean(UCIQE)
        # self.UIQM = np.mean(UIQM)
        self.my_print("UCIQE:{}".format(self.UCIQE))
        # self.my_print("UIQM:{}".format(self.UIQM))


    def SaveModel(self, epoch):
        Student_global_G_path = os.path.join(self.model_save_dir, '{}-global_G.ckpt'.format(epoch))
        torch.save(self.student_G.state_dict(), Student_global_G_path)
        # for net, name in zip(self.netAs, self.mapping_layers):
        #     torch.save(net.weight.data,
        #                os.path.join(self.model_save_dir, '{}-'.format(epoch) + name + '_weight.pt'))
        #     torch.save(net.bias.data,
        #                os.path.join(self.model_save_dir, '{}-'.format(epoch) + name + '_bias.pt'))
        for k in range(self.num_recurrent):
            for l in range(self.num_branch):
                torch.save(self.__getattribute__("D_" + str(k + 1) + "_" + str(l + 1)).state_dict(),
                           os.path.join(self.model_save_dir,
                                        ("{}-D_" + str(k + 1) + "_" + str(l + 1) + ".ckpt").format(epoch)))
        self.my_print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    def leDecay(self, epoch):
        if epoch >= self.num_epoch_decay and epoch % self.lr_update_step == 0:
            global_glr = self.global_glr / (2 ** ((epoch - self.num_epoch_decay) // self.lr_update_step) + 1)
            global_dlr = self.global_dlr / (2 ** ((epoch - self.num_epoch_decay) // self.lr_update_step) + 1)
            self.update_lr(global_glr, global_dlr)
            self.my_print('Decayed learning rates, global_glr: {}. global_dlr: {}.'
                          .format(global_glr, global_dlr))

    def train(self):
        """train model"""
        self.start_epoch = 1
        if self.Super_train_discri_epoch:
            self.restore_Super_model()
        if self.Train_resume:
            self.start_epoch = self.Train_resume
            self.restore_Norm_model(self.Train_resume)
        self.my_print('Start training...')
        self.start_time = time.time()
        for epoch in range(self.start_epoch, self.num_epoch + 1):
            for bz_iter, tensor_dic in enumerate(self.train_loader):
                self.muddy, self.clean = tensor_dic["muddy"], tensor_dic["clean"]
                self.distored_real = self.muddy.to(self.device)
                self.clean_real = self.clean.to(self.device)
                # self.Sample()
                for k in range(self.num_recurrent):
                    self.reset_grad()
                    self.forward()
                    self.backward_D(k)
                    self.reset_grad()
                    self.backward_G(k)
                    self.distored_real = self.distored_step2  # 这一次循环的输出是下一次recurrent的输入
                self.SaveLog(bz_iter, epoch)
            self.TrainImage(epoch)
            self.TestImage(epoch)
            self.SaveModel(epoch)
            self.leDecay(epoch)
        self.print_log.close()