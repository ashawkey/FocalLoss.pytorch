import os
import cv2
import glob
import time
import tqdm
import tensorboardX
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from . io import logger
from . utils import DelayedKeyboardInterrupt, summary
from . init import init_weights



class Trainer(object):
    """Base trainer class. 
    """

    def __init__(self, 
                 conf,
                 model, 
                 optimizer, 
                 lr_scheduler, 
                 objective, 
                 dataloaders,
                 logger,
                 metrics=[],
                 input_shape=None,
                 use_checkpoint="latest",
                 restart=False,
                 max_keep_ckpt=1,
                 eval_set="test",
                 test_set="test",
                 eval_interval=1,
                 report_step_interval=200,
                 max_eval_step=None,
                 use_parallel=False,
                 use_tqdm=True,
                 use_tensorboardX=True,
                 weight_init_function=init_weights,
                 ):
        
        self.conf = conf
        self.device = conf.device
        self.workspace_path = conf.workspace

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.objective = objective
        self.dataloaders = dataloaders
        self.metrics = metrics
        self.log = logger
        self.use_checkpoint = use_checkpoint
        self.max_keep_ckpt = max_keep_ckpt
        self.restart = restart
        self.eval_set = eval_set
        self.test_set = test_set
        self.eval_interval = eval_interval
        self.report_step_interval = report_step_interval
        self.max_eval_step = max_eval_step
        
        self.use_parallel = use_parallel
        self.use_tqdm = use_tqdm
        self.use_tensorboardX = use_tensorboardX

        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.log.info(f'Time stamp is {self.time_stamp}')

        self.model.to(self.device)

        if input_shape is not None:
            summary(self.model, input_shape, logger=self.log)

        if self.use_parallel:
            self.model = nn.DataParallel(self.model)

        if weight_init_function is not None:
            self.model.apply(weight_init_function)

        self.log.info(f'Number of model parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "StepLoss": [],
            "EpochLoss": [],
            "EvalResults": [],
            "Checkpoints": [],
            "BestResult": None,
            }

        if self.workspace_path is not None:
            os.makedirs(self.workspace_path, exist_ok=True)
            if self.use_checkpoint == "latest":
                self.log.info("Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "scratch":
                self.log.info("Train from scratch")
            elif self.use_checkpoint == "best":
                self.log.info("Loading best checkpoint ...")
                model_name = type(self.model).__name__
                ckpt_path = os.path.join(self.workspace_path, 'checkpoints')
                best_path = f"{ckpt_path}/{model_name}_best.pth.tar"
                self.load_checkpoint(best_path)
            else: # path to ckpt
                self.log.info(f"Loading checkpoint {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)


    ### ---------------------------------------------------
    ### example step function for input segmentation task
    ### ---------------------------------------------------


    def train_step(self, data):
        input, truth = data["input"], data["truth"]

        output = self.model(input)
        loss = self.objective(output, truth)
        pred = output.detach().cpu().numpy().argmax(axis=1)

        data["output"] = output
        data["pred"] = pred
        data["loss"] = loss

        return data

    def eval_step(self, data):	
        input, truth = data["input"], data["truth"]

        output = self.model(input)
        pred = output.detach().cpu().numpy().argmax(axis=1)

        data["output"] = output
        data["pred"] = pred

        return data

    def predict_step(self, data):	
        input = data["input"]
        
        output = self.model(input)
        pred = output.detach().cpu().numpy().argmax(axis=1)

        data["output"] = output
        data["pred"] = pred

        return data

    ### ---------------------------------------------------

    def train(self, max_epochs=None):
        """
        do the training process for max_epochs.
        """
        if max_epochs is None:
            max_epochs = self.conf.max_epochs

        if self.use_tensorboardX:
            logdir = os.path.join(self.workspace_path, "run", self.time_stamp)
            self.writer = tensorboardX.SummaryWriter(logdir)
        
        for epoch in range(self.epoch, max_epochs+1):
            self.epoch = epoch
            
            self.train_one_epoch()

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(self.eval_set)

                if self.workspace_path is not None:
                    self.save_checkpoint()

        if self.use_tensorboardX:
            self.writer.close()

        self.log.info("Finished Training.")

    def evaluate(self, eval_set=None):
        """
        final evaluate at the best epoch.
        """
        eval_set = self.eval_set if eval_set is None else eval_set
        self.log.info(f"Evaluate at the best epoch on {eval_set} set...")

        # load model
        model_name = type(self.model).__name__
        ckpt_path = os.path.join(self.workspace_path, 'checkpoints')
        best_path = f"{ckpt_path}/{model_name}_best.pth.tar"
        if not os.path.exists(best_path):
            self.log.error(f"Best checkpoint not found! {best_path}")
            raise FileNotFoundError
        self.load_checkpoint(best_path)

        # turn off logging to tensorboardX
        self.use_tensorboardX = False
        self.evaluate_one_epoch(eval_set)

    def get_time(self):
        if torch.cuda.is_available(): 
            torch.cuda.synchronize()
        return time.time()

    def prepare_data(self, data):
        if isinstance(data, list) or isinstance(data, tuple):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device)
        else: # is_tensor
            data = data.to(self.device)

        return data

    def profile(self, steps=1):
        """
        ```bash
        python -m torch.utils.bottleneck train.py 
        ```
        """
        self.log.log(f"==> Start Profiling for {steps} steps.")

        self.model.train()
        for metric in self.metrics:
            metric.clear()

        start_time = self.get_time()

        data_time = 0
        forward_time = 0
        backward_time = 0
        metric_time = 0

        for i in range(steps):

            data_start_time = self.get_time()
            data = next(iter(self.dataloaders["train"]))
            data = self.prepare_data(data)
            data_time += self.get_time() - data_start_time
            
            forward_start_time = self.get_time()
            data = self.train_step(data)
            truth, pred, loss = data["truth"], data["pred"], data["loss"]
            forward_time += self.get_time() - forward_start_time
            
            backward_start_time = self.get_time()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            backward_time += self.get_time() - backward_start_time
            
            metric_start_time = self.get_time()
            for metric in self.metrics:
                metric.update(pred, truth)
            metric_time += self.get_time() - metric_start_time

        self.lr_scheduler.step()

        end_time = self.get_time()

        self.log.log(f"==> Finished Profiling for {steps} steps, time={end_time-start_time:.4f}({data_time:.4f} + {forward_time:.4f} + {backward_time:.4f} + {metric_time:.4f})")


    def train_one_epoch(self):
        self.log.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']} ...")

        for metric in self.metrics:
            metric.clear()
        total_loss = []
        self.model.train()

        pbar = self.dataloaders["train"]
        if self.use_tqdm:
            pbar = tqdm.tqdm(pbar)

        self.local_step = 0
        epoch_start_time = self.get_time()

        for data in pbar:
            start_time = self.get_time()
            self.local_step += 1
            self.global_step += 1
            
            data = self.prepare_data(data)

            data = self.train_step(data)
            truth, pred, loss = data["truth"], data["pred"], data["loss"]

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            for metric in self.metrics:
                metric.update(pred, truth)
                if self.use_tensorboardX:
                    metric.write(self.writer, self.global_step, prefix="train")
                    
            if self.use_tensorboardX:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)

            total_loss.append(loss.item())
            total_time = self.get_time() - start_time

            if self.report_step_interval > 0 and self.local_step % self.report_step_interval == 0:
                self.log.log1(f"step={self.epoch}/{self.local_step}, loss={loss.item():.4f}, time={total_time:.2f}")
                for metric in self.metrics:
                    self.log.log1(metric.report())
                    metric.clear()

        if self.report_step_interval < 0:
            for metric in self.metrics:
                self.log.log1(metric.report())
                metric.clear()

        self.lr_scheduler.step()
        epoch_end_time = self.get_time()
        average_loss = np.mean(total_loss)
        self.stats["StepLoss"].extend(total_loss)
        self.stats["EpochLoss"].append(average_loss)

        self.log.log(f"==> Finished Epoch {self.epoch}, average_loss={average_loss:.4f}, time={epoch_end_time-epoch_start_time:.4f}")


    def evaluate_one_epoch(self, eval_set):
        self.log.log(f"++> Evaluate at epoch {self.epoch} ...")

        for metric in self.metrics:
            metric.clear()
        self.model.eval()

        pbar = self.dataloaders[eval_set]
        if self.use_tqdm:
            pbar = tqdm.tqdm(pbar)

        epoch_start_time = self.get_time()

        with torch.no_grad():
            self.local_step = 0
            start_time = self.get_time()
            
            for data in pbar:    
                self.local_step += 1
                
                if self.max_eval_step is not None and self.local_step > self.max_eval_step:
                    break
                
                data = self.prepare_data(data)
                data = self.eval_step(data)
                pred, truth = data["pred"], data["truth"]
                
                for metric in self.metrics:
                    metric.update(pred, truth)

            total_time = self.get_time() - start_time
            self.log.log1(f"total_time={total_time:.2f}")
            
            self.stats["EvalResults"].append(self.metrics[0].measure())

            for metric in self.metrics:
                self.log.log1(metric.report())
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        epoch_end_time = self.get_time()
        self.log.log(f"++> Evaluate Finished. time={epoch_end_time-epoch_start_time:.4f}")


    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""
        with DelayedKeyboardInterrupt():
            model_name = type(self.model).__name__
            ckpt_path = os.path.join(self.workspace_path, 'checkpoints')
            file_path = f"{ckpt_path}/{model_name}_ep{self.epoch:04d}.pth.tar"
            best_path = f"{ckpt_path}/{model_name}_best.pth.tar"
            os.makedirs(ckpt_path, exist_ok=True)

            self.stats["Checkpoints"].append(file_path)

            if len(self.stats["Checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["Checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
                    self.log.info(f"Removed old checkpoint {old_ckpt}")

            state = {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'model_name': model_name,
                'model': self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'stats' : self.stats,
            }
            
            torch.save(state, file_path)
            self.log.info(f"Saved checkpoint {self.epoch} successfully.")
            
            if self.stats["EvalResults"] is not None:
                if self.stats["BestResult"] is None or self.metrics[0].better(self.stats["EvalResults"][-1], self.stats["BestResult"]):
                    self.stats["BestResult"] = self.stats["EvalResults"][-1]
                    torch.save(state, best_path)
                    self.log.info(f"Saved Best checkpoint.")
            

    def load_checkpoint(self, checkpoint=None):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the model at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        model_name = type(self.model).__name__
        
        ckpt_path = os.path.join(self.workspace_path, 'checkpoints')
        
        if checkpoint is None:
            # Load most recent checkpoint            
            checkpoint_list = sorted(glob.glob(f'{ckpt_path}/{model_name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                self.log.info("No checkpoint found, model randomly initialized.")
                return False
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = f'{ckpt_path}/{model_name}_ep{checkpoint:04d}.pth.tar'
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            self.log.error("load_checkpoint: Invalid argument")
            raise TypeError

        checkpoint_dict = torch.load(checkpoint_path)

        #assert model_name == checkpoint_dict['model_name'], 'network is not of correct type.'

        self.model.load_state_dict(checkpoint_dict['model'])
        if not self.restart:
            self.log.info("Loading epoch and other status...")
            self.epoch = checkpoint_dict['epoch'] + 1
            self.global_step = checkpoint_dict['global_step']
            self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
            self.lr_scheduler.last_epoch = checkpoint_dict['epoch'] 
            self.stats = checkpoint_dict['stats']
        else:
            self.log.info("Only loading model parameters.")
        
        self.log.info("Checkpoint Loaded Successfully.")
        return True


