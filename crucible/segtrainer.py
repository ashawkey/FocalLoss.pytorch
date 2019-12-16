from . trainer import *
from . vision import plot_images

import skimage.transform as skt


class SegTrainer3D(Trainer):

    def resize(self, image, new_shape, interpolation=1):
        B, C, H, W, D = image.shape
        image = image.reshape(-1, H, W, D).transpose(1,2,3,0)
        image = skt.resize(image, new_shape, order=interpolation, mode='constant', cval=0, clip=True, anti_aliasing=False)
        image = image.transpose(3,0,1,2).reshape(B, C, new_shape[0], new_shape[1], new_shape[2])
        return image

    def pad(self, image, new_shape, border_mode="constant", value=0):
        '''
        image: [B, C, H, W, D]
        new_shape: [H, W, D]
        '''
        axes_not_pad = len(image.shape) - len(new_shape)

        old_shape = np.array(image.shape[-len(new_shape):])
        new_shape = np.array([max(new_shape[i], old_shape[i]) for i in range(len(new_shape))])

        difference = new_shape - old_shape
        pad_below = difference // 2
        pad_above = difference - pad_below

        pad_list = [[0, 0]] * axes_not_pad + [list(i) for i in zip(pad_below, pad_above)]

        if border_mode == 'reflect':
            res = np.pad(image, pad_list, border_mode)
        elif border_mode == 'constant':
            res = np.pad(image, pad_list, border_mode, constant_values=value)
        else:
            raise NotImplementedError
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)

        return res, slicer
    
    def mirroring_forward(self, x):
        y = self.model(x)
        for dims in [[0],[1],[2],[0,1],[1,2],[0,2],[0,1,2]]:
            y = y + torch.flip(self.model(torch.flip(x, dims)), dims)
        y = y / 8
        return y

    def eval_step(self, data):
        """
        image: [B, C, H, W]
        """
        image = data["image"].numpy()
        
        if self.conf.eval_type == 'whole':
            B, C, H, W, D = image.shape
            resized_image = self.resize(image, self.conf.patch_size, 3)

            resized_image = torch.from_numpy(resized_image).float().to(self.device)
            pred = self.model(resized_image) 
            pred = pred.detach().cpu().numpy()
            
            pred = self.resize(pred, (H, W, D), 3) 
            pred = pred.argmax(1)

        elif self.conf.eval_type == 'sliding_window':
            
            image, slicer = self.pad(image, self.conf.patch_size)

            B, C, H, W, D = image.shape
            # pad to at least patch_size
            pred_sum = np.zeros((B, self.conf.num_classes, H, W, D))
            pred_cnt = np.zeros((1, 1, H, W, D))
            # slide window
            ph, pw, pd = self.conf.patch_size
            sh, sw, sd = self.conf.patch_stride
            for h in range(0, H, sh):
                for w in range(0, W, sw):
                    for d in range(0, D, sd):
                        hh = min(h + ph, H)
                        ww = min(w + pw, W)
                        dd = min(d + pd, D)
                        h = hh - ph
                        w = ww - pw
                        d = dd - pd

                        patch = image[:, :, h:hh, w:ww, d:dd] # [B, C, ph, pw, pd]
                        patch = torch.from_numpy(patch).float().to(self.device)

                        # not enough
                        pred = self.model(patch)
                        #pred = self.mirroring_forward(patch)

                        pred = pred.detach().cpu().numpy()

                        pred_sum[:, :, h:hh, w:ww, d:dd] += pred
                        pred_cnt[0, 0, h:hh, w:ww, d:dd] += 1

            pred = pred_sum / pred_cnt
            pred = pred[:, :, slicer[2], slicer[3], slicer[4]]
            pred = pred.argmax(1) # [B, H, W, D]

        else:
            raise NotImplementedError

        data["pred"] = pred

        return data
        

    def evaluate(self, 
                 eval_set=None,
                 save_snap=True,
                 save_image=False,
                 save_image_folder=None,
                 show_image=False,
                 ):

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
            self.log.error(f"Best checkpoint not found at {best_path}, load by default.")
            self.load_checkpoint()
        else:
            self.load_checkpoint(best_path)

        # turn off logging to tensorboardX
        self.use_tensorboardX = False
        self.evaluate_one_epoch(eval_set, save_snap, save_image, save_image_folder, show_image)

    def evaluate_one_epoch(self, 
                           eval_set,
                           save_snap=False,
                           save_image=False,
                           save_image_folder=None,
                           show_image=False,
                           ):
        self.log.log(f"++> Evaluate at epoch {self.epoch} ...")

        for metric in self.metrics:
            metric.clear()

        self.model.eval()

        pbar = self.dataloaders[eval_set]
        if self.use_tqdm:
            pbar = tqdm.tqdm(pbar)

        epoch_start_time = self.get_time()

        if save_image:
            if save_image_folder is None:
                save_image_folder = 'evaluation_' + self.time_stamp
            save_image_folder = os.path.join(self.workspace_path, save_image_folder)
            os.makedirs(save_image_folder, exist_ok=True)

        with torch.no_grad():
            self.local_step = 0
            start_time = self.get_time()
            
            for data in pbar:    
                self.local_step += 1
                                
                if self.max_eval_step is not None and self.local_step > self.max_eval_step:
                    break

                data = self.eval_step(data)
                pred, mask = data["pred"], data["mask"]

                for metric in self.metrics:
                    metric.update(pred, mask)
                
                if show_image:
                    batch_size = pred.shape[0]
                    f = plt.figure()
                    ax0 = f.add_subplot(121)
                    ax1 = f.add_subplot(122)
                    for batch in range(batch_size):
                        ax0.imshow(pred[batch])
                        ax1.imshow(mask[batch])
                        plt.show()
                
                if save_image:
                    batch_size = pred.shape[0]
                    for batch in range(batch_size):
                        if 'name' in data:
                            name = data['name'][batch] + '.npy'
                        else:
                            name = str(self.local_step) + '_' + str(batch) + '.npy'
                        
                        np.save(os.path.join(save_image_folder, name), pred[batch])
                        self.log.info(f"Saved image {name} at {save_image_folder}.")
                    
            
            total_time = self.get_time() - start_time
            self.log.log1(f"total_time={total_time:.2f}")
            
            self.stats["EvalResults"].append(self.metrics[0].measure())

            if save_snap and self.use_tensorboardX:
                # only save first batch first layer
                self.writer.add_image("evaluate/image", data["image"][0, :, :, 0], self.epoch)
                self.writer.add_image("evaluate/pred", np.expand_dims(pred[0, :, :, 0], 0), self.epoch)
                self.writer.add_image("evaluate/mask", np.expand_dims(mask[0, :, :, 0], 0), self.epoch)

            for metric in self.metrics:
                self.log.log1(metric.report())
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        epoch_end_time = self.get_time()
        self.log.log(f"++> Evaluate Finished. time={epoch_end_time-epoch_start_time:.4f}")

    def predict(self, eval_set='test', save_image=True, save_image_folder=None, show_image=False):
        self.log.log(f"++> Predict at epoch {self.epoch} ...")

        self.model.eval()

        pbar = self.dataloaders[eval_set]
        if self.use_tqdm:
            pbar = tqdm.tqdm(pbar)

        epoch_start_time = self.get_time()

        if save_image:
            if save_image_folder is None:
                save_image_folder = 'prediction_' + self.time_stamp
            save_image_folder = os.path.join(self.workspace_path, save_image_folder)
            os.makedirs(save_image_folder, exist_ok=True)

        with torch.no_grad():
            self.local_step = 0
            start_time = self.get_time()
            
            for data in pbar:    
                self.local_step += 1
                                
                data = self.eval_step(data)
                pred = data["pred"]
                
                if save_image:
                    batch_size = pred.shape[0]
                    for batch in range(batch_size):
                        if 'name' in data:
                            name = data['name'][batch] + '.npy'
                        else:
                            name = str(self.local_step) + '_' + str(batch) + '.npy'
                        
                        np.save(os.path.join(save_image_folder, name), pred[batch])
                        self.log.info(f"Saved image {name} at {save_image_folder}.")
                    
            
            total_time = self.get_time() - start_time
            self.log.log1(f"total_time={total_time:.2f}")
            
        epoch_end_time = self.get_time()
        self.log.log(f"++> Predict Finished. time={epoch_end_time-epoch_start_time:.4f}")