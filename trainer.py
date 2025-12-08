import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
import sklearn
from sklearn.mixture import GaussianMixture
import utils
import numpy as np


class Trainer:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.net = kwargs['net']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = self.args.writer
        self.logger = self.args.logger
        #self.criterion = ASDLoss().to(self.args.device)
        self.transform = kwargs['transform']



    def train(self, train_loader):

        self.net.eval()
        with torch.no_grad():
            self.net.set_c(train_loader) # 设定超球面中心

        # self.test(save=False)
        model_dir = os.path.join(self.writer.log_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        epochs = self.args.epochs
        valid_every_epochs = self.args.valid_every_epochs
        early_stop_epochs = self.args.early_stop_epochs
        start_valid_epoch = self.args.start_valid_epoch
        num_steps = len(train_loader)
        self.sum_train_steps = 0
        self.sum_valid_steps = 0
        best_metric = 0
        no_better_epoch = 0

        for epoch in range(0, epochs + 1):
            
            
            # train
            sum_total_loss = 0
            sum_svdd_loss = 0
            sum_classification_loss = 0
            sum_contrastive_loss = 0
            self.net.train()
            train_bar = tqdm(train_loader, total=num_steps, desc=f'Epoch-{epoch}')

            # 混合训练
            for (x_wavs, x_mels, labels) in train_bar:
                # forward
                x_wavs, x_mels = x_wavs.float().to(self.args.device), x_mels.float().to(self.args.device)
                labels = labels.reshape(-1).long().to(self.args.device)
                x_mels = x_mels.unsqueeze(1)
                x_wavs = x_wavs.unsqueeze(1)

                # svdd_loss = self.net.compute_oneclass_loss(x_mels, labels)  # 单分类超球面损失
                #svdd_loss = self.net.compute_loss(x_mels,x_wavs,labels) # 多分类超球面损失， 根据对应的标签计算对应的超球面损失
                
                svdd_loss,_ = self.net.compute_soft_svdd_loss(x_mels,x_wavs,labels) # 软边界多分类超球面损失
                #classfication_loss,_ = self.net.compute_classification_loss(x_mels,x_wavs,labels) # 根据对应的标签计算对应的分类损失
                # loss_sup = self.net.compute_supcon_loss(x_mels,labels) # svdd对比损失

                loss = svdd_loss #+ classfication_loss #+ loss_sup #

                train_bar.set_postfix(loss=f'{loss.item():.5f}')
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                # visualization
                self.writer.add_scalar(f'total_loss', loss.item(), self.sum_train_steps)
                self.writer.add_scalar(f'svdd_loss', svdd_loss.item(), self.sum_train_steps)
                #self.writer.add_scalar(f'classification_loss', classfication_loss.item(), self.sum_train_steps)
                # self.writer.add_scalar(f'contrastive_loss', loss_sup.item(), self.sum_train_steps)

                sum_total_loss += loss.item()
                sum_svdd_loss += svdd_loss.item()
                #sum_classification_loss += classfication_loss.item()
                #sum_contrastive_loss += loss_sup.item()
                self.sum_train_steps += 1


            avg_total_loss = sum_total_loss / num_steps
            avg_svdd_loss = sum_svdd_loss / num_steps
            avg_classification_loss = sum_classification_loss / num_steps
            avg_contrastive_loss = sum_contrastive_loss / num_steps

            if self.scheduler is not None and epoch >= self.args.start_scheduler_epoch:
                self.scheduler.step()
            self.logger.info(f'Epoch-{epoch}\ttotal_loss:{avg_total_loss:.5f}, '
                             f'svdd_loss:{avg_svdd_loss:.5f}, '
                             f'classification_loss:{avg_classification_loss:.5f},'
                             f'contrastive_loss:{avg_contrastive_loss:.5f}')


            # valid
            if (epoch - start_valid_epoch) % valid_every_epochs == 0 and epoch >= start_valid_epoch:
                avg_auc, avg_pauc = self.test(save=False, gmm_n=False)
                self.writer.add_scalar(f'auc', avg_auc, epoch)
                self.writer.add_scalar(f'pauc', avg_pauc, epoch)
                if avg_auc + avg_pauc >= best_metric:
                    no_better_epoch = 0
                    best_metric = avg_auc + avg_pauc
                    best_model_path = os.path.join(model_dir, 'best_checkpoint.pth.tar')
                    utils.save_model_state_dict(best_model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)
                    self.logger.info(f'Best epoch now is: {epoch:4d}')
                else:
                    # early stop
                    no_better_epoch += 1
                    if no_better_epoch > early_stop_epochs > 0: break

            # save last 10 epoch state dict
            if epoch >= self.args.start_save_model_epochs:
                if (epoch - self.args.start_save_model_epochs) % self.args.save_model_interval_epochs == 0:
                    model_path = os.path.join(model_dir, f'{epoch}_checkpoint.pth.tar')
                    utils.save_model_state_dict(model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)

    def test(self, save=False, gmm_n=False):
        """
            gmm_n if set as number, using GMM estimator (n_components of GMM = gmm_n)
            if gmm_n = sub_center(arcface), using weight vector of arcface as the mean vector of GMM
        """
        csv_lines = []
        sum_auc, sum_pauc, num = 0, 0, 0
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        
        os.makedirs(result_dir, exist_ok=True)
        self.net.eval()
        net = self.net.module if self.args.dp else self.net
        print('\n' + '=' * 20)

        for index, (target_dir, train_dir) in enumerate(zip(sorted(self.args.valid_dirs), sorted(self.args.train_dirs))):
            machine_type = target_dir.split('/')[-2]
            # result csv
            csv_lines.append([machine_type])
            csv_lines.append(['id', 'AUC', 'pAUC'])
            performance = []
            # get machine list
            machine_id_list = utils.get_machine_id_list(target_dir)
            for id_str in machine_id_list:
                meta = machine_type + '-' + id_str
                label = self.args.meta2label[meta]
                test_files, y_true = utils.create_test_file_list(target_dir, id_str, dir_name='test')
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{id_str}.csv')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
               
                for file_idx, file_path in enumerate(test_files):
                    x_wav, x_mel, label = self.transform(file_path)
                    x_wav, x_mel = x_wav.unsqueeze(0).float().to(self.args.device), x_mel.unsqueeze(0).float().to(
                        self.args.device)
                    label = torch.tensor([label]).long().to(self.args.device)
                    with torch.no_grad():
                        #predict_ids, feature = net(x_wav, x_mel, label)
                        #predict_ids = net.compute_classification_anomaly_score(x_mel.unsqueeze(1),x_wav.unsqueeze(1))
                        svdd_anomaly_score = net.compute_anomaly_score(x_mel.unsqueeze(1),x_wav.unsqueeze(1),label) # 多分类SVDD异常分数
                        #svdd_anomaly_score = net.compute_oneclass_anomaly_score(x_mel.unsqueeze(1),label) # 单分类SVDD异常分数
                        #svdd_anomaly_score = net.soft_boundary_anomaly_score(x_mel.unsqueeze(1),x_wav.unsqueeze(1),label) # 软边界多分类SVDD异常分数
                        #svdd_anomaly_score = net.compute_anomaly_score_with_classification_weight(x_mel.unsqueeze(1),x_wav.unsqueeze(1),label) # Multi-Class SVDD score(Soft)


                   
                    #probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                    svdd_anomaly_score = svdd_anomaly_score.squeeze().cpu().numpy()
                    #y_pred[file_idx] = probs[label] + svdd_anomaly_score #结合分类和SVDD算异常分数
                    y_pred[file_idx] =  svdd_anomaly_score # 仅用SVDD来算异常分数
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                if save:
                    utils.save_csv(csv_path, anomaly_score_list)
                    
                # compute auc and pAuc
                max_fpr = 0.1
                auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
                performance.append([auc, p_auc])

            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            self.logger.info(f'{machine_type}\t\tAUC: {mean_auc*100:.3f}\tpAUC: {mean_p_auc*100:.3f}')
            csv_lines.append(['Average'] + list(averaged_performance))
            sum_auc += mean_auc
            sum_pauc += mean_p_auc
            num += 1
        avg_auc, avg_pauc = sum_auc / num, sum_pauc / num
        csv_lines.append(['Total Average', avg_auc, avg_pauc])
        self.logger.info(f'Total average:\t\tAUC: {avg_auc*100:.3f}\tpAUC: {avg_pauc*100:.3f}')
        result_path = os.path.join(result_dir, 'result.csv')
        if save:
            utils.save_csv(result_path, csv_lines)
        return avg_auc, avg_pauc


