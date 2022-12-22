# -*- coding: utf-8 -*-
# @Time    : 2022/12/21 21:41
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : main.py.py
# @Software: PyCharm

from torch.optim import SGD
from dataset_builder import *
from model import SMP_Unet_meta
from mmengine.runner import Runner
from mmengine.logging import MMLogger

import warnings
import numpy as np
from typing import Sequence
from mmengine.logging import print_log
from prettytable import PrettyTable
from mmeval.metrics import MeanIoU

logger = MMLogger.get_instance('mmseg', log_level='INFO')

random.seed(24)
np.random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.cuda.manual_seed_all(24)
os.environ['PYTHONASHSEED'] = str(24)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class IoUMetric(MeanIoU):
    """A wrapper of ``mmeval.MeanIoU``.
    This wrapper implements the `process` method that parses predictions and
    labels from inputs. This enables ``mmengine.Evaluator`` to handle the data
    flow of different tasks through a unified interface.
    In addition, this wrapper also implements the ``evaluate`` method that
    parses metric results and print pretty tabel of metrics per class.
    Args:
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.
        **kwargs: Keyword parameters passed to :class:`mmeval.MeanIoU`.
    """

    def __init__(self, dist_backend='torch_cuda', **kwargs):
        iou_metrics = kwargs.pop('iou_metrics', None)
        if iou_metrics is not None:
            warnings.warn(
                'DeprecationWarning: The `iou_metrics` parameter of '
                '`IoUMetric` is deprecated, defaults return all metrics now!')
        collect_device = kwargs.pop('collect_device', None)

        if collect_device is not None:
            warnings.warn(
                'DeprecationWarning: The `collect_device` parameter of '
                '`IoUMetric` is deprecated, use `dist_backend` instead.')

        # Changes the default value of `classwise_results` to True.
        super().__init__(classwise_results=True,
                         dist_backend=dist_backend,
                         **kwargs)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.
        Parse predictions and labels from ``data_samples`` and invoke
        ``self.add``.
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        predictions, labels = [], []
        for data_sample in data_samples:
            pred_label = torch.argmax(data_sample[1], dim=1)
            label = torch.argmax(data_sample[2], dim=1)
            predictions.append(pred_label)
            labels.append(label)

        self.add(predictions, labels)

    def evaluate(self, *args, **kwargs):
        """Returns metric results and print pretty tabel of metrics per class.
        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)
        self.reset()

        classwise_results = metric_results['classwise_results']
        del metric_results['classwise_results']

        # Pretty table of the metric results per class.
        summary_table = PrettyTable()
        summary_table.add_column('Class', self.dataset_meta['classes'])
        for key, value in classwise_results.items():
            value = np.round(value * 100, 2)
            summary_table.add_column(key, value)

        print_log('per class results:', logger='current')
        print_log('\n' + summary_table.get_string(), logger='current')

        # Multiply value by 100 to convert to percentage and rounding.
        evaluate_results = {
            k: round(v * 100, 2) for k, v in metric_results.items()}
        return evaluate_results


# class MeanIoU_torch(BaseMetric):
#
#     def process(self, data_batch, data_samples: List[torch.Tensor]):
#         output = data_samples[1]
#         target = data_samples[2]
#         # save the middle result of a batch to `self.results`
#         self.results.append({'output': torch.argmax(output, dim=1),
#                              'target': torch.argmax(target, dim=1)})
#
#     def compute_metrics(self, results: list) -> dict:
#         output = [item['output'] for item in results]
#         target = [item['target'] for item in results]
#         output = [o.tolist() for o in output]
#         target = [t.tolist() for t in target]
#         with torch.no_grad():
#             mean_iou = JaccardIndex(num_classes=13)(torch.tensor(output), torch.tensor(target))
#             con_matrix = ConfusionMatrix(num_classes=13)(torch.tensor(output), torch.tensor(target)) \
#                          + torch.tensor(1e-8)
#             PA = torch.diag(con_matrix).sum() / con_matrix.sum()
#             CPA = torch.diag(con_matrix) / torch.sum(con_matrix, dim=1)
#             mean_ACC = torch.mean(CPA)
#             intersection = torch.diag(con_matrix)  # 取对角元素的值，返回列表
#             union = torch.sum(con_matrix, dim=1) + torch.sum(con_matrix, dim=0) - torch.diag(
#                 con_matrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
#             IoU = intersection / union
#
#         logger.info(CPA)
#         logger.info(IoU)
#         return dict(mean_iou=mean_iou,
#                     pixelAccuracy=PA, meanPixelAccuracy=mean_ACC,
#                     )


use_metadata = True

path_data = "/root/autodl-tmp/FLAIR_Project/flair_dataset/"
path_metadata_file = "/root/autodl-tmp/FLAIR_Project/metadata/flair-one_metadata.json"

path_data = "/root/autodl-tmp/FLAIR_Project/flair_dataset/"
path_metadata_file = "/root/autodl-tmp/FLAIR_Project/metadata/flair-one_metadata.json"


dict_train, dict_val, dict_test = step_loading(path_data, path_metadata_file, use_metadata=use_metadata)
dataModule = DataModule(dict_train, dict_val, dict_test, num_workers=8, batch_size=32, use_metadata=use_metadata,
                        use_augmentations=train_pipeline)
dataModule.setup(stage='fit')
train_dataloader, val_dataloader, predict_dataloader = \
    dataModule.train_dataloader(), dataModule.val_dataloader(), dataModule.predict_dataloader()

model = SMP_Unet_meta(n_channels=5, n_classes=13, use_metadata=use_metadata)

param_scheduler = [
    # 在 [0, 100) 迭代时使用线性学习率
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=True,
         begin=0,
         end=10),
    # 在 [100, 900) 迭代时使用余弦学习率
    dict(type='CosineAnnealingLR',
         T_max=90,
         by_epoch=True,
         begin=10,
         end=100)
]

runner = Runner(
    # the model used for training and validation.
    # Needs to meet specific interface requirements
    model=model,
    # working directory which saves training logs and weight files
    work_dir='./work_dir',
    # train dataloader needs to meet the PyTorch data loader protocol
    train_dataloader=train_dataloader,
    # optimize wrapper for optimization with additional features like
    # AMP, gradient accumulation, etc
    optim_wrapper=dict(type='AmpOptimWrapper', optimizer=dict(type=SGD, lr=0.001, momentum=0.9, weight_decay=0.0005)),
    param_scheduler=param_scheduler,
    # training coins for specifying training epoch's, verification intervals, etc
    train_cfg=dict(by_epoch=True, max_epochs=100, val_interval=1),
    # validation dataloader also needs to meet the PyTorch data loader protocol
    val_dataloader=val_dataloader,
    # validation configs for specifying additional parameters required for validation
    val_cfg=dict(),
    # validation evaluator. The default one is used here
    val_evaluator=dict(type=IoUMetric),
    # launcher='pytorch',
    visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend', save_dir='temp_dir'),
                                                     dict(type='WandbVisBackend', init_kwargs=
                                                     dict(project='FLAIR_Project'))]),
    default_hooks = dict(checkpoint=dict(type='CheckpointHook', save_best='mean_iou', rule='greater',
                                         interval=1, max_keep_ckpts=5))
)
runner.train()
