#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# public
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# private
from src import helper
from src.eval import Evaluator
from src.models import LSP
from src.datamodule import DataModule


class LSPTrainer(object):
    """docstring for LSPTrainer"""
    def __init__(self, config, **kwargs):
        super(LSPTrainer, self).__init__()
        self.config = config
        self.update_config(**kwargs)
        self.initialize()

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def initialize(self):
        # logger
        self.logger = helper.init_logger(self.config)
        self.logger.info('Logger initialized.')
        self.wandb_logger = WandbLogger(
            name=self.config.NAME
            , save_dir=self.config.LOG_PATH
            , offline=self.config.offline
            , project=self.config.PROJECT
            , log_model=self.config.log_model
            , entity=self.config.ENTITY
            , save_code=False
            , mode=self.config.wandb_mode
            )
        self.wandb_logger.experiment.config.update(self.config)
        # model
        self.model = LSP(self.config)
        match self.config.train_mode:
            case 'base':
                pass
            case 'finetune':
                self.logger.info('Finetuned model loading...')
                self.model = self.model.load_from_checkpoint(
                    self.config.FT_CKPT_PATH
                    , config=self.config
                    )
        # datamodule
        self.dm = DataModule(self.config)
        # callbacks
        if self.config.monitor == 'val_p1':
            filename = '{epoch}-{step}-{val_p1:.4f}'
        elif self.config.monitor == 'val_f10':
            filename = '{epoch}-{step}-{val_f10:.4f}'
        else:
            raise NotImplementedError
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.CKPT_PATH
            , filename=filename
            , monitor=self.config.monitor
            , mode='max'
            , verbose=True
            , save_last=True
            , save_top_k=1
            )
        early_stop_callback = EarlyStopping(
            monitor=self.config.monitor
            , min_delta=.0
            , patience=self.config.patience
            , verbose=True
            , mode='max'
            )
        # trainer
        self.trainer = L.Trainer(
            accelerator=self.config.accelerator
            , precision=self.config.precision
            , logger = self.wandb_logger
            , callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar()]
            , fast_dev_run=self.config.fast_dev_run
            , max_epochs=self.config.max_epochs
            , val_check_interval=self.config.val_check_interval
            , check_val_every_n_epoch=self.config.check_val_every_n_epoch
            , enable_checkpointing=True
            , enable_progress_bar=True
            , gradient_clip_val=self.config.gradient_clip_val
            , deterministic=True
            , inference_mode=True
            , profiler=self.config.profiler if self.config.profiler else None
            , num_sanity_val_steps=2
            )

    def train(self):
        self.logger.info('*Configurations:*')
        for k, v in self.config.__dict__.items():
            self.logger.info(f'\t{k}: {v}')
        # training
        self.logger.info("Start training...")
        self.trainer.fit(
            model=self.model
            , datamodule=self.dm
            , ckpt_path= 'last' if self.config.load_ckpt else None
            )
        # testing
        self.logger.info("Start testing...")
        predict_dict = self.predict(ckpt_path='best')
        # evaluation
        eva = Evaluator(predict_dict)
        self.logger.info(eva.info)
        # save results
        helper.save_pickle(predict_dict, self.config.RESULTS_PKL)
        self.logger.info('Results saved as {}.'.format(self.config.RESULTS_PKL))
        # upload to wandb
        self.update_wandb(predict_dict)
        self.logger.info('Done.')

    def validate(self, ckpt_path=None):
        self.trainer.validate(
            model=self.model
            , datamodule=self.dm
            , ckpt_path=ckpt_path
            , verbose=True
            )

    def predict(self, ckpt_path=None):
        # list of dict
        outputs_list = self.trainer.predict(
            model=self.model
            , datamodule=self.dm
            , ckpt_path=ckpt_path
            , return_predictions=True
            )
        # formatting
        outputs_dict = dict()
        for k in outputs_list[0]:
            outputs_dict[k] = [d[k] for d in outputs_list]
        
        # result_path = './res/results/lsp/genesis/wswitch/best/base/gpt2/'
        # job = 'lsp-genesis-wswitch-best-base-gpt2-0'
        # outputs_dict = pd.read_csv(
        #     os.path.join(result_path, f'{job}.csv')
        #     , delimiter=','
        #     )
        # outputs_dict['subs'] = [s.split(',') for s in outputs_dict.subs]
        # outputs_dict['subs_'] = [s.split(',') for s in outputs_dict.subs_]

        # postprocessing
        self.logger.info("Start postprocessing...")
        outputs_dict['clean_subs_'] = helper.postprocess(
            target=outputs_dict['target']
            , pos=outputs_dict['pos']
            , subs_=outputs_dict['subs_']
            )
        # ranking
        self.logger.info("Start ranking...")
        outputs_dict['rank_subs_'] = helper.rank(
            target=outputs_dict['target']
            , position=outputs_dict['position']
            , context=outputs_dict['context']
            , subs_=outputs_dict['clean_subs_']
            , config=self.config
            )
        return outputs_dict

    def update_wandb(self, predict_dict):
        for k in predict_dict:
            dtype = type(predict_dict[k][0])
            if dtype == str:
                pass
            elif dtype == int:
                predict_dict[k] = [str(v) for v in predict_dict[k]]
            elif dtype == list:
                predict_dict[k] = [','.join(v) for v in predict_dict[k]]
            else:
                raise NotImplementedError
        self.wandb_logger.log_text(
            key='test'
            , columns=list(predict_dict.keys())
            , data=[[predict_dict[k][i] for k in predict_dict] for i in range(self.config.predict_size)]
            )