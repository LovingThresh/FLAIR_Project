# -*- coding: utf-8 -*-
# @Time    : 2022/12/21 21:41
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : main.py.py
# @Software: PyCharm

from typing import List


from torch.optim import SGD
from dataset_builder import *
from model import SMP_Unet_meta
from torchmetrics import JaccardIndex, ConfusionMatrix
from mmengine.runner import Runner
from mmengine.logging import MMLogger
from mmengine.evaluator import BaseMetric


logger = MMLogger.get_instance('mmseg', log_level='INFO')

random.seed(24)
np.random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.cuda.manual_seed_all(24)
os.environ['PYTHONASHSEED'] = str(24)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MeanIoU_torch(BaseMetric):

    def process(self, data_batch, data_samples: List[torch.Tensor]):
        output = data_samples[1]
        target = data_samples[2]
        # save the middle result of a batch to `self.results`
        self.results.append({'output': torch.argmax(output, dim=1),
                             'target': torch.argmax(target, dim=1)})

    def compute_metrics(self, results: list) -> dict:
        output = [item['output'] for item in results]
        target = [item['target'] for item in results]
        output = [o.tolist() for o in output]
        target = [t.tolist() for t in target]

        con_matrix = ConfusionMatrix(num_classes=13)(torch.tensor(output).int(), torch.tensor(target).int()) \
                     + torch.tensor(1e-8)
        PA = torch.diag(con_matrix).sum() / con_matrix.sum()
        CPA = torch.diag(con_matrix) / torch.sum(con_matrix, dim=1)
        mean_ACC = torch.mean(CPA)
        intersection = torch.diag(con_matrix)  # 取对角元素的值，返回列表
        union = torch.sum(con_matrix, dim=1) + torch.sum(con_matrix, dim=0) - torch.diag(
            con_matrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union

        logger.info(CPA)
        logger.info(IoU)
        return dict(mean_iou=JaccardIndex(num_classes=13)(torch.tensor(output).int(), torch.tensor(target).int()),
                    pixelAccuracy=PA, meanPixelAccuracy=mean_ACC,
                    # classPixelAccuracy=CPA, IntersectionOverUnion=IoU
                    )


use_metadata = False
path_data = "/root/autodl-tmp/flair-one-starting-kit/toy_dataset_flair-one/"
path_metadata_file = "/root/autodl-tmp/flair-one-starting-kit/metadata/flair-one_TOY_metadata.json"
dict_train, dict_val, dict_test = step_loading(path_data, path_metadata_file, use_metadata=use_metadata)
dataModule = DataModule(dict_train, dict_val, dict_test, num_workers=1, batch_size=2, use_metadata=use_metadata,
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
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9, weight_decay=0.0005)),
    param_scheduler=param_scheduler,
    # training coins for specifying training epoch's, verification intervals, etc
    train_cfg=dict(by_epoch=True, max_epochs=20, val_interval=1),
    # validation dataloader also needs to meet the PyTorch data loader protocol
    val_dataloader=val_dataloader,
    # validation configs for specifying additional parameters required for validation
    val_cfg=dict(),
    # validation evaluator. The default one is used here
    val_evaluator=dict(type=MeanIoU_torch),
    visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend', save_dir='temp_dir'),
                                                     dict(type='WandbVisBackend', init_kwargs=
                                                     dict(project='FLAIR_Project', notes=None,
                                                          tags=None, group='Segmentation'))])
)
runner.train()
