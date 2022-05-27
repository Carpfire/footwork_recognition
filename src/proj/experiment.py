from cv2 import repeat
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from vpd_fencing_depr.models.module import FCNet
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc
import torch
import wandb
import sys
import os 
import random
from IPython.display import HTML
import numpy as np
from tqdm import tqdm

class Experiment:

    def __init__(self, criterion, optimizer, model, epochs, device, train_loader, valid_loader, exp_name, model_dir=None, test_loader=None):
        self.training_loss = []
        self.validation_loss = []
        self.criterion = criterion
        self.optimizer = optimizer 
        self.model = model
        self.epochs = epochs
        self.device = device
        self.train_loader = train_loader
        self.val_loader = valid_loader
        self.test_loader = test_loader
        self.exp_name = exp_name
        self.model_dir = model_dir
        self.label_converter = {y:x for x,y in self.val_loader.dataset.label_dict.items()}



    def get_loss_and_correct(self, batch):
        data, target = batch
        target = target.long().to(self.device)
        data = (data[0].to(self.device), data[1].to(self.device))
        pred = self.model(data)
        classes = pred.max(dim = 1)[1]
        loss = self.criterion(pred, target)
        size = len(target) 
        correct = (classes == target).sum()
        return loss, correct, size

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        pass

    def epoch(self):
        '''Calls Train and Validate/ Called By run()'''
        train_loss = 0.
        val_loss = 0.
        train_acc = []
        val_acc = []
        self.model.train()
        for batch in self.train_loader:
            loss, batch_acc = self.train(batch)
            train_loss += loss
            train_acc.append(batch_acc)
        self.model.eval()
        for batch in self.val_loader:
            loss, batch_acc = self.validate(batch)
            val_loss += loss
            val_acc.append(batch_acc)
        epoch_train_acc = torch.mean(torch.Tensor(train_acc))
        epoch_val_acc = torch.mean(torch.Tensor(val_acc))
        mean_train_loss = train_loss/len(self.train_loader)
        mean_val_loss = val_loss/len(self.val_loader)

        return epoch_train_acc, epoch_val_acc, mean_train_loss, mean_val_loss

    def train(self, batch):

        loss, correct, size = self.get_loss_and_correct(batch)
        self.step(loss)

        return loss, correct/size
    
    def validate(self, batch):
        '''Validation Loop'''
        with torch.no_grad():
            loss, correct, size = self.get_loss_and_correct(batch)

        return loss, correct/size
    def results(self, return_inputs = False):
        predictions = []
        ground_truth = []
        if return_inputs:
            input_data = []
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()
            for batch in self.val_loader:
                data, target = batch
                if return_inputs:
                    input_data.extend([(d, l) for d,l  in zip(data[0], data[1])])
                data = (data[0].to(self.device), data[1].to(self.device))
                pred = self.model(data)
                pred = pred.max(dim=1)[1].detach().cpu()
                target_list = [int(val.item()) for val in target]
                ground_truth.extend(target_list)
                pred_list = [p.item() for p in pred]
                predictions.extend(pred_list)
        if return_inputs:
            return torch.Tensor(ground_truth), torch.Tensor(predictions), np.array(input_data)
        else: return torch.Tensor(ground_truth), torch.Tensor(predictions)
    def score_report(self):

        gt, pred = self.results()
        target_names = self.val_loader.dataset.label_dict.keys()
        print(classification_report(gt, pred, target_names=target_names))

    def confusion_matrix(self):

        gt, pred = self.results()
        
        gt = [self.label_converter[lab.item()] for lab in gt]
        pred = [self.label_converter[lab.item()] for lab in pred]
        disp = ConfusionMatrixDisplay.from_predictions(gt, pred, xticks_rotation=45.)
        
    def incorrect_predictions(self):
        gt, pred, inputs = self.results(return_inputs=True)
        inds = torch.where(gt != pred)[0]
        return gt[inds], pred[inds], inputs[inds]
    
    def view_preds(self, decoder_path, correct = True):
        if correct:
            gt, pred, inputs = self.correct_predictions() 
        else: 
            gt, pred, inputs = self.incorrect_predictions()
        ind = random.randint(0, len(gt)-1)
        choice_gt, choice_pred = gt[ind], pred[ind]
        pose, length = inputs[ind]
        pose_seq = pose[:int(length.item()), :]
        return self.view_action(pose_seq, choice_gt, choice_pred, decoder_path)

    def correct_predictions(self):
        gt, pred, inputs = self.results(return_inputs=True)
        inds = torch.where(gt == pred)[0]
        return gt[inds], pred[inds], inputs[inds]


    def view_action(self, pose_seq, gt_label, pred_label, decoder_path):
        
        decoder = FCNet(26, [128, 128], 2*26, dropout = 0)
        decoder.load_state_dict(torch.load(decoder_path))
        decoder.to(self.device)
        decoder.eval() 
        fig, ax = plt.subplots()
        ax.set_title(f"Pred: {self.label_converter[int(pred_label.item())]}, Ground Truth: {self.label_converter[int(gt_label.item())]}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        circles = ax.scatter([],[],s=4)
        def data_gen():
            for pose in pose_seq:
                inpose = pose.reshape(2, 26)[0]
                decoded_pose = decoder(inpose.to(self.device)).detach().cpu().numpy()
                yield np.c_[decoded_pose[0:26:2], -decoded_pose[1:26:2]]
        def draw(data):
            circles.set_offsets(data)
            return circles
        rc('animation', html='jshtml')
        anim = FuncAnimation(fig, draw, data_gen)

        return anim

    def save_model(self, name):
        self.name = name
        model_path = os.path.join(self.model_dir, ''.join([name, '.pt']))
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_name):
        self.name = model_name
        model_path = os.path.join(self.model_dir, model_name)
        self.model.load_state_dict(torch.load(model_path))

    def run(self, model_name = None, WANDB=True):
        '''This is where the looping logic will run'''
        if WANDB:
            run = wandb.init(project=self.exp_name, entity='carpfire')
            wandb.config = {
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epochs': self.epochs,
                'batch_size':self.train_loader.batch_size
            }

        self.model.to(self.device)
        self.criterion.to(self.device)
        pbar = tqdm(range(self.epochs), file=sys.stdout)
        # if WANDB:
        #     artifact = wandb.Artifact('footwork_dataset', type='dataset')
        #     artifact.add_file(self.train_loader.dataset.ann_file)
        #     run.log_artifact(artifact)
        
        def refresh_loss(a, va, l, vl):
            pbar.set_description(f'tl {l:.2f}, ta {a:.2f} vl {vl:.2f}, va {va:.2f}')
            pbar.refresh()

        for e in pbar:
            epoch_metrics = self.epoch()
            if WANDB:
                wandb.log({
                    'validation_acc':epoch_metrics[1],
                    'train_acc':epoch_metrics[0],
                    'validation_loss':epoch_metrics[3],
                    'train_loss':epoch_metrics[2]
                })
                wandb.watch(self.model)
            refresh_loss(*epoch_metrics)
        
        if model_name:
            self.save_model(model_name)
            if WANDB:
                artifact = wandb.Artifact(model_name, type='model')
                print(artifact)
                artifact.add_file(os.path.join(self.model_dir, ''.join([model_name, '.pt'])))
                run.log_artifact(artifact)
        if WANDB:
            wandb.finish()
        
        return self.model
