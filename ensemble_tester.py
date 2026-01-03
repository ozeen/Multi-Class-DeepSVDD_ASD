import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
import sklearn
from sklearn.mixture import GaussianMixture
import utils
import numpy as np


class EnsembleTester:
    def __init__(self, *args, **kwargs):
        """
        kwargs 约定：
            args:       训练/测试超参数配置（原来的 self.args）
            nets:       已经加载好权重的模型列表 [net1, net2, ...]
            transform:  数据预处理/特征提取函数，与原代码一致
        """
        self.args = kwargs['args']
        self.nets = kwargs['nets']          # <<< 新增：多个模型
        self.writer = self.args.writer
        self.logger = self.args.logger
        self.transform = kwargs['transform']

    def test(self, save=False, gmm_n=False):
        """
            gmm_n if set as number, using GMM estimator (n_components of GMM = gmm_n)
            if gmm_n = sub_center(arcface), using weight vector of arcface as the mean vector of GMM

            与原 test 基本一致；不同点：
            - 使用 self.nets 中的多个模型分别计算 anomaly score
            - 对多个模型的 anomaly score 取平均作为最终分数
        """
        csv_lines = []
        sum_auc, sum_pauc, num = 0, 0, 0
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)

        # 所有模型设为 eval，并根据 dp 取出底层 net
        base_nets = []
        for net in self.nets:
            net.eval()
            base_nets.append(net)

        print('\n' + '=' * 20)
        for index, (target_dir, train_dir) in enumerate(
                zip(sorted(self.args.valid_dirs), sorted(self.args.train_dirs))):
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
                    x_wav, x_mel, label_tmp = self.transform(file_path)
                    x_wav = x_wav.unsqueeze(0).float().to(self.args.device)
                    x_mel = x_mel.unsqueeze(0).float().to(self.args.device)
                    # 这里 label 仍按原逻辑转换为 tensor
                    label_tensor = torch.tensor([label_tmp]).long().to(self.args.device)

                    with torch.no_grad():
                        # === 关键修改：对多个模型分别算分数，然后取平均 ===
                        svdd_scores = []

                        for net in base_nets:
                            svdd_anomaly_score = net.compute_anomaly_score(
                                x_mel.unsqueeze(1),  # [B, 1, ...]
                                x_wav.unsqueeze(1),  # [B, 1, ...]
                                label_tensor
                            )
                            svdd_scores.append(svdd_anomaly_score)

                        # stack 后在模型维度求均值，得到 ensemble 分数
                        svdd_anomaly_score = torch.stack(svdd_scores, dim=0).mean(dim=0)


                    svdd_anomaly_score = svdd_anomaly_score.squeeze().cpu().numpy()
                    # 仅用 ensemble 后的 SVDD 分数作为异常分数
                    y_pred[file_idx] = svdd_anomaly_score
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







