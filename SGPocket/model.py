from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch_geometric as pyg
from torchmetrics.functional.classification import (binary_accuracy,
                                                    binary_auroc,
                                                    binary_f1_score,
                                                    binary_precision,
                                                    binary_recall, stat_scores)

import SGPocket.utilities.pocket_segmentation as pock_seg
from SGPocket.networks.network import SGCN

NB_RES_NODE_FTS = 20
SH_ORDER = 5


class DiceLoss(torch.nn.Module):
    # The Dice Loss
    # https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor,
                smooth: float = 1) -> torch.Tensor:
        """Compute the Dice loss

        Args:
            inputs (torch.Tensor): Network outputs
            targets (torch.Tensor): Ground Thruth
            smooth (float, optional): Smooth parameter for the loss. Defaults to 1.

        Returns:
            torch.Tensor: The loss
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class Model(pl.LightningModule):
    def __init__(self,
                 lr: float,
                 weight_decay: float,
                 hidden_channels: List[int],
                 mlp_channels: List[int],
                 dropout: float = 0.0,
                 non_linearity: str = 'none'):
        """Constructor

        Args:
            lr (float): The learning rate
            weight_decay (float): The weight decay
            hidden_channels (List[int]): List of hidden channels sizes
            mlp_channels (List[int]): List of MLP sizes
            dropout (float, optional): Dropout. Defaults to 0.0.
            non_linearity (str, optional): Non-lineary used. Defaults to 'none'.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.training_step_outputs = {'loss': [],
                                      'preds': [],
                                      'targets': []
                                      }

        self.val_step_outputs = {'loss': [],
                                 'preds': [],
                                 'targets': []
                                 }

        self.test_step_outputs = {'loss': [],
                                  'preds': [],
                                  'targets': []
                                  }

        self.net = SGCN(NB_RES_NODE_FTS,
                        hidden_channels,
                        mlp_channels,
                        dropout,
                        SH_ORDER,
                        non_linearity)

        self.loss_funct = DiceLoss()

    def forward(self, data: pyg.data.Batch) -> torch.Tensor:
        """Apply the model to a batch

        Args:
            data (pyg.data.Batch): A batch of graphs

        Returns:
            torch.Tensor: The prediction for all nodes of batch graphs
        """
        pred_res = self.net(data.x, data.edge_index, data.edge_attr)
        return torch.sigmoid(pred_res)

    def _common_step(self,
                     batch: pyg.data.Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Common step for Train, Val and Test
        Args:
            batch (pyg.data.Batch): A Batch
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (loss,
                predicted score, real affinities)
        """
        y_pred = self(batch)

        loss = self.loss_funct(y_pred, batch.y)
        return loss, y_pred, batch.y

    def training_step(self,
                      batch: pyg.data.Batch,
                      batch_idx: int) -> torch.Tensor:
        """Training step
        Args:
            batch (pyg.data.Batch): A Batch
            batch_idx (int): The batch idx
        Returns:
            torch.Tensor: loss
        """
        loss, preds, targets = self._common_step(batch)
        self.training_step_outputs['loss'].append(loss)
        self.training_step_outputs['preds'].append(preds.detach())
        self.training_step_outputs['targets'].append(targets.detach())
        return loss

    def validation_step(self,
                        batch: pyg.data.Batch,
                        batch_idx: int) -> torch.Tensor:
        """Validation step
        Args:
            batch (pyg.data.Batch): A Batch
            batch_idx (int): The batch idx
        Returns:
            torch.Tensor: loss
        """
        loss, preds, targets = self._common_step(batch)
        self.val_step_outputs['loss'].append(loss)
        self.val_step_outputs['preds'].append(preds)
        self.val_step_outputs['targets'].append(targets)
        return loss

    def test_step(self,
                  batch: pyg.data.Batch,
                  batch_idx: int) -> torch.Tensor:
        """Testing step
        Args:
            batch (pyg.data.Batch): A Batch
            batch_idx (int): The batch idx
        Returns:
            torch.Tensor: loss
        """
        loss, preds, targets = self._common_step(batch)
        self.test_step_outputs['loss'].append(loss)
        self.test_step_outputs['preds'].append(preds)
        self.test_step_outputs['targets'].append(targets)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        return {"optimizer": optimizer}

    def on_train_epoch_end(self):
        self.common_epoch_end(self.training_step_outputs, 'train')
        self.training_step_outputs.clear()
        self.training_step_outputs = {'loss': [],
                                      'preds': [],
                                      'targets': []
                                      }

    def on_validation_epoch_end(self):
        self.common_epoch_end(self.val_step_outputs, 'val')
        self.val_step_outputs.clear()
        self.val_step_outputs = {'loss': [],
                                 'preds': [],
                                 'targets': []
                                 }

    def on_test_epoch_end(self):
        self.common_epoch_end(self.test_step_outputs, 'test')

        self.test_step_outputs.clear()
        self.test_step_outputs = {'loss': [],
                                  'preds': [],
                                  'targets': []
                                  }

    def common_epoch_end(self, outputs: Dict, stage: str):
        loss_batched = torch.stack(outputs['loss'])
        preds_batched = torch.concat([x.reshape(-1) for x in outputs['preds']])
        targets_batched = torch.concat(
            [x.reshape(-1) for x in outputs['targets']])
        avg_loss = loss_batched.mean()
        bin_accuracy = binary_accuracy(preds_batched, targets_batched)
        f1_score = binary_f1_score(preds_batched, targets_batched)
        confusion_matrix = stat_scores(
            preds_batched, targets_batched, task='binary')
        auroc = binary_auroc(preds_batched, targets_batched.int())
        precision = binary_precision(preds_batched, targets_batched)
        recall = binary_recall(preds_batched, targets_batched)
        tp = confusion_matrix[0].float()
        fp = confusion_matrix[1].float()
        tn = confusion_matrix[2].float()
        fn = confusion_matrix[3].float()
        metrics_dict = {
            "ep_end_{}/loss".format(stage): avg_loss,
            "ep_end_{}/bin_accuracy".format(stage): bin_accuracy,
            "ep_end_{}/f1_score".format(stage): f1_score,
            "ep_end_{}/auroc".format(stage): auroc,
            "ep_end_{}/precision".format(stage): precision,
            "ep_end_{}/recall".format(stage): recall,
            "ep_end_{}/tp".format(stage): tp,
            "ep_end_{}/fp".format(stage): fp,
            "ep_end_{}/tn".format(stage): tn,
            "ep_end_{}/fn".format(stage): fn
        }
        self.log_dict(metrics_dict, sync_dist=True)

    def predict(self, data: pyg.data.Batch) -> torch.Tensor:
        """Predicts node class

        Args:
            data (pyg.data.Batch): A Batch of graphs

        Returns:
            torch.Tensor: Nodes prediction
        """
        self.eval()
        with torch.no_grad():
            y_pred = self(data)
        return y_pred

    def predict_and_extract(self,
                            data: pyg.data.Batch,
                            threshold: float,
                            dbscan_eps: float) -> Tuple[torch.Tensor, Dict]:
        """Predicts and extracts pockets

        Args:
            data (pyg.data.Batch): A Batch of graphs
            threshold (float): Threshold to classify nodes
            dbscan_eps (float): DBScan epsilon

        Returns:
            Tuple[torch.Tensor, Dict]: Cluster id for each node, Nb of nodes in each cluster
        """
        y_pred = self.predict(data)
        segmented_labels, dict_lab_freq = pock_seg.clusterize(
            y_pred, data.pos, th=threshold, dbscan_eps=dbscan_eps)
        return segmented_labels, dict_lab_freq

    def predict_extract_and_assess(self,
                                   data: pyg.data.Batch,
                                   threshold: float,
                                   dbscan_eps: float) -> Tuple[float, float, torch.Tensor, Dict]:
        """Predicts, extracts and assess

        Args:
            data (pyg.data.Batch): A Batch of graphs
            threshold (float): Threshold to classify nodes
            dbscan_eps (float): DBScan epsilon

        Returns:
            Tuple[float, float, torch.Tensor, Dict]: DCC, Mean DCC, Cluster id for each node, Nb of nodes in each cluster
        """
        segmented_labels, dict_lab_freq = self.predict_and_extract(
            data, threshold, dbscan_eps)
        if segmented_labels is not None:
            dcc_min, dcc_mean = pock_seg.get_dcc(
                segmented_labels, dict_lab_freq, data.pos, data.sites_center)
            return dcc_min, dcc_mean, segmented_labels, dict_lab_freq,
        else:
            return None, None, None, None
