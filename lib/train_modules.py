# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import torch
import pytorch_lightning as pl
from lib.loss import InfoNCE, KDSP, NST, Attention, CC, PKT, RKD, FitNets, AB, SoftTarget, Logits
import torchmetrics


class MultimodalContrastiveLearningModule(pl.LightningModule):
    def __init__(self, modality_to_encoder, source_modality, target_modalities):
        """
        modality_to_encoder = {
                'imu': imu_encoder,
                'text': text_encoder,
                'video': video_encoder,
                'audio': audio_encoder,
            }
        """
        super().__init__()

        self.source_modality = source_modality
        self.target_modalities = target_modalities
        self.list_modalities = modality_to_encoder.keys()

        #self.loss = torch.nn.CrossEntropyLoss()
        self.loss1 = InfoNCE(symmetric_loss=True)
        #self.loss2 = KDSP()
        self.loss3 = NST()
        #self.loss4 = Attention()
        #self.loss5 = CC(gamma =0.4, P_order = 2)
        #self.loss6 = PKT()
        #self.loss7 = RKD(w_dist=25.0, w_angle=50.0)
        #self.loss8 = FitNets()
        #self.loss9 = AB(margin=2.0)
        #self.loss10 = SoftTarget(T=4.0)
        #self.loss11 = Logits()

        if "imu" in self.list_modalities:
            self.imu_encoder = modality_to_encoder["imu"] #Student Model

        if "text" in self.list_modalities:
            self.text_encoder = modality_to_encoder["text"] #Teacher Model

        if "video" in self.list_modalities:
            self.video_encoder = modality_to_encoder["video"]

        if "audio" in self.list_modalities:
            self.audio_encoder = modality_to_encoder["audio"]

    def forward(self, batch):
        # x_imu: (batch_size x 6 x window_size)
        # x_narration: [ str ] with len == batch_size
        # y_*: B x size_embeddings
        """
        if len(batch["video"]) != len(batch["narration"]) or len(batch["video"]) != len(batch["imu"]):
            print("Weird!")
            min_size = min(min(len(batch["video"]), len(batch["narration"])), len(batch["imu"]))
            batch["imu"] = batch["imu"][:min_size]
            batch["video"] = batch["video"][:min_size]
            batch["audio"] = batch["audio"][:min_size]
        """

        out = {}

        if "imu" in self.list_modalities:
            x_imu = batch["imu"]
            y_imu = self.imu_encoder(x_imu)
            out["imu"] = y_imu

        if "text" in self.list_modalities:
            x_narration = batch["narration"]
            y_narration = self.text_encoder.get_text_embeddings(x_narration)
            out["text"] = y_narration

        if "video" in self.list_modalities:
            x_video = batch["video"]
            y_video = self.video_encoder.get_video_embeddings(x_video)
            out["video"] = y_video

        if "audio" in self.list_modalities:
            x_audio = batch["audio"]
            y_audio = self.audio_encoder(x_audio)
            out["audio"] = y_audio

        return out

    def training_step(self, batch, batch_idx: int):
        return self._shared_eval(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx: int):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int):
        # y: {modality[str]: y_*} where y_*: B x size_embeddings
        y = self(batch)

        # Prepare metrics computation
        y_query_modality = y[self.source_modality]
        loss_output = 0.0
        metrics = {}

        # Compute metrics for source modality <> each target modality
        for target_modality in self.target_modalities:
            y_key_modality = y[target_modality]
            lambda_kd = 1.0
            #celoss = self.loss(y_query_modality, y_key_modality)
            info_nceloss = self.loss1(query=y_query_modality, positive_key=y_key_modality)
            #sploss = (self.loss2(fm_s= y_query_modality, fm_t = y_key_modality) + self.loss2(fm_s= y_query_modality, fm_t = y_key_modality) + self.loss2(fm_s= y_query_modality, fm_t = y_key_modality))/3.0 * lambda_kd
            nstloss = self.loss3(fm_s = y_query_modality, fm_t= y_key_modality) * lambda_kd
            #atloss = (self.loss4(fm_s= y_query_modality, fm_t = y_key_modality) + self.loss4(fm_s= y_query_modality, fm_t = y_key_modality) + self.loss4(fm_s= y_query_modality, fm_t = y_key_modality))/3.0 * lambda_kd
            #ccloss = self.loss5(feat_s = y_query_modality, feat_t = y_key_modality) * lambda_kd
            #pktloss = self.loss6(feat_s = y_query_modality, feat_t = y_key_modality) * lambda_kd
            #rkdloss = self.loss7(feat_s = y_query_modality, feat_t = y_key_modality) * lambda_kd
            #fitnetloss = self.loss8(fm_s = y_query_modality, fm_t= y_key_modality) * lambda_kd
            #abloss = (self.loss9(fm_s= y_query_modality, fm_t = y_key_modality) + self.loss9(fm_s= y_query_modality, fm_t = y_key_modality) + self.loss9(fm_s= y_query_modality, fm_t = y_key_modality))/3.0 * lambda_kd
            #stloss = self.loss10(out_s = y_query_modality, out_t = y_key_modality) * lambda_kd
            #logitloss = self.loss11(out_s = y_query_modality, out_t = y_key_modality) * lambda_kd
            s2t_loss = info_nceloss + nstloss
            loss_output += s2t_loss
            s_t_accuracy, t_s_accuracy = evaluate_batch_similarity(
                y_query_modality, y_key_modality, device=self.device
            )

            str_s2t = "{source_modality_initial}2{target_modality_initial}".format(
                source_modality_initial=self.source_modality[0],
                target_modality_initial=target_modality[0],
            )
            str_t2s = "{target_modality_initial}2{source_modality_initial}".format(
                target_modality_initial=target_modality[0],
                source_modality_initial=self.source_modality[0],
            )
            metrics[f"{str_s2t}_accuracy"] = s_t_accuracy
            metrics[f"{str_t2s}_accuracy"] = t_s_accuracy

        # Save and log metrics
        metrics["test_loss"] = loss_output
        self.log_dict(metrics, logger=True)
        return metrics

    def predict_step(self, batch, batch_idx: int):
        return self(batch)

    def _shared_eval(self, batch, batch_idx: int, prefix: str):
        # y: {modality[str]: y_*} where y_*: B x size_embeddings
        y = self(batch)

        # Use NCE loss
        y_query_modality = y[self.source_modality]
        loss_output = 0.0

        # Compute loss for source modality <> each target modality
        for target_modality in self.target_modalities:
            y_key_modality = y[target_modality]
            lambda_kd = 1.0
            #celoss = self.loss(y_query_modality, y_key_modality)
            info_nceloss = self.loss1(query=y_query_modality, positive_key=y_key_modality)
            #sploss = (self.loss2(fm_s= y_query_modality, fm_t = y_key_modality) + self.loss2(fm_s= y_query_modality, fm_t = y_key_modality) + self.loss2(fm_s= y_query_modality, fm_t = y_key_modality))/3.0 * lambda_kd
            nstloss = self.loss3(fm_s = y_query_modality, fm_t= y_key_modality) * lambda_kd
            #atloss = (self.loss4(fm_s= y_query_modality, fm_t = y_key_modality) + self.loss4(fm_s= y_query_modality, fm_t = y_key_modality) + self.loss4(fm_s= y_query_modality, fm_t = y_key_modality))/3.0 * lambda_kd
            #ccloss = self.loss5(feat_s = y_query_modality, feat_t = y_key_modality) * lambda_kd
            #pktloss = self.loss6(feat_s = y_query_modality, feat_t = y_key_modality) * lambda_kd
            #fitnetloss = self.loss8(fm_s = y_query_modality, fm_t= y_key_modality) * lambda_kd
            #abloss = (self.loss9(fm_s= y_query_modality, fm_t = y_key_modality) + self.loss9(fm_s= y_query_modality, fm_t = y_key_modality) + self.loss9(fm_s= y_query_modality, fm_t = y_key_modality))/3.0 * lambda_kd
            #stloss = self.loss10(out_s = y_query_modality, out_t = y_key_modality) * lambda_kd
            #logitloss = self.loss11(out_s = y_query_modality, out_t = y_key_modality) * lambda_kd
            s2t_loss = info_nceloss + nstloss         
            str_s2t = "{source_modality_initial}2{target_modality_initial}".format(
                source_modality_initial=self.source_modality[0],
                target_modality_initial=target_modality[0],
            )
            self.log(f"{prefix}_{str_s2t}_loss", s2t_loss, logger=True, sync_dist=True)
            loss_output += s2t_loss

        self.log(f"{prefix}_loss", loss_output, logger=True, sync_dist=True)
        return loss_output

    def configure_optimizers(self):
        #return torch.optim.SGD(self.parameters(), lr = 2e-4, momentum = 0.9, weight_decay = 1e-4, nesterov = True)
        return torch.optim.Adam(self.parameters(), lr=2e-4)#2e-4 7e-7


def evaluate_batch_similarity(source_embeddings, target_embeddings, device):
    """
    Given a batch matrix (size B) of paired embeddings,
    measure the accuracy of the predictions by checking nearest the neighbors
    """
    #  Compute similarity
    s = torch.nn.functional.normalize(source_embeddings, dim=1)
    t = torch.nn.functional.normalize(target_embeddings, dim=1)

    # similarities: B x B
    similarities = torch.mm(s, t.transpose(0, 1))

    # pred: 1 x B (ideally [0, 1, 2, 3, ..., B])
    s_t_pred = torch.argmax(similarities, dim=1)
    t_s_pred = torch.argmax(similarities, dim=0)
    B = len(s_t_pred)
    s_t_accuracy = sum(s_t_pred == torch.arange(B, device=device)) / B
    t_s_accuracy = sum(t_s_pred == torch.arange(B, device=device)) / B
    return s_t_accuracy, t_s_accuracy


class ClassificationModule(pl.LightningModule):
    def __init__(self, model):
        """
        Encoder + Head
        """
        super().__init__()

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model = model
        self.accuracy_train = torchmetrics.Accuracy(task="multiclass", num_classes=4)
        self.accuracy_valid = torchmetrics.Accuracy(task="multiclass", num_classes=4)
        self.accuracy_test = torchmetrics.Accuracy(task="multiclass", num_classes=4)
        self.f1_test = torchmetrics.F1Score(task="multiclass", num_classes=4, average="macro")

    def forward(self, batch):
        # x_imu: (batch_size x 6 x window_size)
        # x_narration: [ str ] with len == batch_size
        # y_*: B x size_embeddings
        """
        in: batch_size x 6 x window_size
        out: batch_size x 1
        """
        return self.model(batch)

    def training_step(self, batch, batch_idx: int):
        return self._shared_eval(batch, batch_idx, "train")

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log("train_acc_epoch", self.accuracy_train)

    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log("val_acc_epoch", self.accuracy_valid)

    def test_epoch_end(self, outs):
        # log epoch metric
        self.log("test_acc_epoch", self.accuracy_test)
        self.log("test_f1_epoch", self.f1_test)

    def validation_step(self, batch, batch_idx: int):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int):
        return self._shared_eval(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx: int):
        return self(batch)

    def _shared_eval(self, batch, batch_idx: int, prefix: str):
        x, y = batch
        y_hat = self(x)
        loss_output = self.loss_fn(y_hat, y)
        if prefix == "train":
            self.accuracy_train(y_hat, y)
            self.log(f"{prefix}_acc_step", self.accuracy_train, logger=True)
        if prefix == "val":
            self.accuracy_valid(y_hat, y)
            self.log(f"{prefix}_acc_step", self.accuracy_valid, logger=True)
        if prefix == "test":
            self.accuracy_test(y_hat, y)
            self.f1_test(y_hat, y)
            self.log(f"{prefix}_acc_step", self.accuracy_test, logger=True)
            self.log(f"{prefix}_f1_step", self.f1_test, logger=True)
        self.log(f"{prefix}_loss", loss_output, logger=True)
        return loss_output

    def configure_optimizers(self):
        #return torch.optim.SGD(self.parameters(), lr = 2e-4, momentum = 0.9, weight_decay = 1e-4, nesterov = True)#5e-4
        return torch.optim.Adam(self.parameters(), lr=5e-4)#5e-4