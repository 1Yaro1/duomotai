import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler
import torch.nn.functional as F
from ts_benchmark.baselines.MindTS.models.MindTS_model import MINDTSModel
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, anomaly_detection_multi_data_provider, anomaly_detection_timeMMD_data_provider
from ts_benchmark.baselines.utils import train_val_split
from ts_benchmark.baselines.MindTS.utils.tools import EarlyStopping, adjust_learning_rate
from torch import optim
import time
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MINDTS_BASED_HYPER_PARAMS = {
    "top_k": 3,
    "enc_in": 4,
    "dec_in": 4,
    "c_out": 4,
    "e_layers": 1,
    "d_layers": 1,
    "d_model": 256,
    "d_ff": 256,
    "embed": "timeF",
    "freq": "h",
    "lradj": "type1",
    "moving_avg": 25,
    "num_kernels": 6,
    "factor": 1,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 72,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 6,
    "patch_size": 6,
    "stride": 6,
    "dropout": 0.1,
    "batch_size": 16,
    "lr": 0.0001,
    "num_epochs": 3,
    "num_workers": 0,
    "loss": "MSE",
    "itr": 1,
    "distil": True,
    "patience": 3,
    "task_name": "anomaly_detection",
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "mem_dim": 32,
    "anomaly_ratio": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    "conv_kernel": [12, 16],
    "use_norm": True,
    "parallel_strategy": "DP",
    "num_epochs": 3,
    "mask_ratio": 0.5,
    "r": 0.5,
    "lamda": 1.0,
    "enc_in_time": 1,
    "lamda1": 1.0,
    "lamda2": 1.0
}

def clip_loss(logits_per_time, logits_per_text):
    labels = torch.arange(logits_per_time.shape[1]).long().to(device)
    total_loss = torch.tensor(0.0).to(device)
    for i in range(logits_per_time.shape[0]):
        total_loss += (F.cross_entropy(logits_per_time[i], labels) + F.cross_entropy(logits_per_text[i], labels)) / 2
    return total_loss

def Bottleneck_loss(total_mask, r, lamda):
    compress_loss, connect_loss = 0., 0.
    for i in range(total_mask.shape[0]):
        temp = total_mask[i]
        compress_loss += (temp * torch.log(temp/(r + 1e-6) + 1e-6) + (1-temp) * torch.log((1-temp)/(1-r+1e-6) + 1e-6)).mean()
        shift1 = temp[1:,:]
        shift2 = temp[:-1,:]
        connect_loss += torch.sum((shift1 - shift2).norm(p=2)) / shift1.flatten().shape[0]    
    connect_loss /= total_mask.shape[0]
    compress_loss /= total_mask.shape[0] 

    mask_loss = compress_loss + lamda * connect_loss
    return mask_loss


class MINDTSConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_MINDTS_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.parallel_strategy not in [None, 'DP']:
            raise ValueError("Invalid value for parallel_strategy. Supported values are 'DP' and None.")

    @property
    def pred_len(self):
        # return self.seq_len
        return 0

    @property
    def learning_rate(self):
        return self.lr
    
    @property
    def model_name(self):
        return "MindTS"
    

class MindTS:
    def __init__(self, **kwargs):
        super(MindTS, self).__init__()
        self.config = MINDTSConfig(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.seq_len = self.config.win_size
        self.lamda1 = self.config.lamda1
        self.lamda2 = self.config.lamda2
        self.log_v2_store = None
        log_mode = getattr(self.config, "log_input_mode", "legacy")
        if log_mode not in {"legacy", "v2"}:
            raise ValueError("log_input_mode must be legacy or v2")
        if log_mode == "v2":
            from ts_benchmark.baselines.MindTS.cmc_log_data import LogV2Store
            self.log_v2_store = LogV2Store(self.config.log_v2_cache_dir, self.config.log_v2_service)
            self.config.log_semantic_dim = self.log_v2_store.semantic_dim
            self.config.log_count_dim = self.log_v2_store.count_dim

    def _multi_data_provider(self, *args, **kwargs):
        if self.log_v2_store is None:
            return anomaly_detection_multi_data_provider(*args, **kwargs)
        from ts_benchmark.baselines.MindTS.cmc_log_data import log_v2_provider
        return log_v2_provider(*args, store=self.log_v2_store, **kwargs)

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {}

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        try:
            freq = pd.infer_freq(train_data.index)
        except Exception as ignore:
            freq = 'S'
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

    def detect_validate(self, valid_data_loader, criterion):
        config = self.config
        total_loss = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for input, _ in valid_data_loader:
                input = input.to(device)

                outputs = self.model(input)

                outputs = outputs[:, :, :]

                outputs = outputs.detach().cpu()
                true = input.detach().cpu()

                loss = criterion(outputs, true).detach().cpu().numpy()

                total_loss.append(loss)  

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss

    def detect_multi_validate(self, valid_data_loader, criterion):
        config = self.config
        total_loss = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for batch_x_time, batch_input_ids, batch_attention_mask, _ in valid_data_loader:
                batch_x_time = batch_x_time.float().to(self.device)
                batch_input_ids = batch_input_ids.float().to(self.device)
                batch_attention_mask = batch_attention_mask.float().to(self.device)
                outputs, logits_per_time, logits_per_text, total_mask = self.model(batch_x_time, batch_input_ids, batch_attention_mask)
                f_dim = -1 if self.config.enc_in == 1 else 0
                outputs = outputs[:, :, f_dim:]

                outputs = outputs.detach().cpu()
                true = batch_x_time.detach().cpu()

                # Reconstruction loss
                loss1 = criterion(outputs, true).detach().cpu().numpy()

                # Comparison Loss
                loss2 = clip_loss(logits_per_time, logits_per_text).detach().cpu().numpy()
                    
                # Bottleneck loss
                loss3 = Bottleneck_loss(total_mask, self.config.r, self.config.lamda).detach().cpu().numpy()

                loss = loss1 + self.lamda1*loss2 + self.lamda2*loss3
                total_loss.append(loss)  

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss
    
    def detect_fit(self, train_data: pd.DataFrame, train_label: pd.DataFrame):
        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        self.model = MINDTSModel(self.config)

        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        config = self.config
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
        )

        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
        )

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stopping = EarlyStopping(patience=config.patience)
        self.model.to(self.device)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        for epoch in range(config.num_epochs):
            self.model.train()
            for i, (input, target) in enumerate(self.train_data_loader):
                optimizer.zero_grad()
                input = input.float().to(self.device)
                outputs = self.model(input)
                outputs = outputs[:, :, :]
                loss = criterion(outputs, input)
                loss.backward()
                optimizer.step()
            valid_loss = self.detect_validate(self.valid_data_loader, criterion)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break

            adjust_learning_rate(optimizer, epoch + 1, config)


    def detect_multi_fit(self, train_data: pd.DataFrame, train_text: pd.DataFrame, train_label: pd.DataFrame):
        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        self.model = MINDTSModel(self.config)

        config = self.config
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        train_data_text, valid_text = train_val_split(train_text, 0.8, None)
        self.scaler.fit(train_data_value.values)

        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        train_data_text = pd.DataFrame(
            train_data_text,
            columns=train_data_text.columns,
            index=train_data_text.index,
        )

        valid_text = pd.DataFrame(
            valid_text,
            columns=valid_text.columns,
            index=valid_text.index,
        )   

        self.valid_data_loader = self._multi_data_provider(
            valid_data,
            valid_text,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
        )

        self.train_data_loader = self._multi_data_provider(
            train_data_value,
            train_data_text,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
        )

        time_now = time.time()

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stopping = EarlyStopping(patience=config.patience)
        self.model.to(self.device)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        for epoch in range(config.num_epochs):
            iter_count = 0
            self.model.train()
            for i, (batch_x_time, batch_input_ids, batch_attention_mask, batch_y) in enumerate(self.train_data_loader):
                iter_count += 1
                train_steps = len(self.train_data_loader)
                optimizer.zero_grad()
                batch_x_time = batch_x_time.float().to(self.device)
                batch_input_ids = batch_input_ids.float().to(self.device)
                batch_attention_mask = batch_attention_mask.float().to(self.device)
                outputs, logits_per_time, logits_per_text, total_mask = self.model(batch_x_time, batch_input_ids, batch_attention_mask)
                f_dim = -1 if self.config.enc_in == 1 else 0
                outputs = outputs[:, :, f_dim:]

                # Reconstruction loss
                loss1 = criterion(outputs, batch_x_time)

                # Comparison Loss
                loss2 = clip_loss(logits_per_time, logits_per_text)

                # Bottleneck loss
                loss3 = Bottleneck_loss(total_mask, self.config.r, self.config.lamda)

                loss = loss1 + self.lamda1*loss2 + self.lamda2*loss3

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1}".format(i + 1, epoch + 1))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((config.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                optimizer.step()
            valid_loss = self.detect_multi_validate(self.valid_data_loader, criterion)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break

            adjust_learning_rate(optimizer, epoch + 1, config)      

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        self.model.to(self.device)
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        attens_energy = []
        test_labels = []

        for i, (batch_x, batch_y) in enumerate(self.thre_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        return test_energy, test_energy

    def detect_multi_score(self, test_data: pd.DataFrame, test_text: pd.DataFrame) -> np.ndarray:
        test_data = pd.DataFrame(
            self.scaler.transform(test_data.values), columns=test_data.columns, index=test_data.index
        )
        test_text = pd.DataFrame(
            test_text.values, columns=test_text.columns, index=test_text.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.thre_loader = self._multi_data_provider(
            test_data,
            test_text,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        self.model.to(self.device)
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        attens_energy = []
        test_labels = []
        for i, (batch_x_time, batch_input_ids, batch_attention_mask, batch_y) in enumerate(self.thre_loader):
            batch_x_time = batch_x_time.float().to(self.device)
            batch_input_ids = batch_input_ids.float().to(self.device)
            batch_attention_mask = batch_attention_mask.float().to(self.device)
            # reconstruction
            outputs, logits_per_time, logits_per_text, total_mask = self.model(batch_x_time, batch_input_ids, batch_attention_mask)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x_time, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        return test_energy, test_energy
    
    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.test_data_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="test",
        )

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        attens_energy = []

        self.model.to(self.device)
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.train_data_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []

        for i, (batch_x, batch_y) in enumerate(self.test_data_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        attens_energy = []
        test_labels = []

        for i, (batch_x, batch_y) in enumerate(self.thre_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy

    def detect_multi_label(self, test_data: pd.DataFrame, test_text: pd.DataFrame) -> np.ndarray:
        test_data = pd.DataFrame(
            self.scaler.transform(test_data.values), columns=test_data.columns, index=test_data.index
        )

        test_text = pd.DataFrame(
            test_text.values, columns=test_text.columns, index=test_text.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.test_data_loader = self._multi_data_provider(
            test_data,
            test_text,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="test",
        )

        self.thre_loader = self._multi_data_provider(
            test_data,
            test_text,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        attens_energy = []

        self.model.to(self.device)
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        with torch.no_grad():
            for i, (batch_x_time, batch_input_ids, batch_attention_mask, batch_y) in enumerate(self.train_data_loader):
                batch_x_time = batch_x_time.float().to(self.device)
                batch_input_ids = batch_input_ids.float().to(self.device)
                batch_attention_mask = batch_attention_mask.float().to(self.device)
                # reconstruction
                outputs, logits_per_time, logits_per_text, total_mask = self.model(batch_x_time, batch_input_ids, batch_attention_mask)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x_time, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x_time, batch_input_ids, batch_attention_mask, batch_y) in enumerate(self.test_data_loader):
            batch_x_time = batch_x_time.float().to(self.device)
            batch_input_ids = batch_input_ids.float().to(self.device)
            batch_attention_mask = batch_attention_mask.float().to(self.device)
            # reconstruction
            outputs, logits_per_time, logits_per_text, total_mask = self.model(batch_x_time, batch_input_ids, batch_attention_mask)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x_time, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        attens_energy = []
        test_labels = []
        for i, (batch_x_time, batch_input_ids, batch_attention_mask, batch_y) in enumerate(self.thre_loader):
            batch_x_time = batch_x_time.float().to(self.device)
            batch_input_ids = batch_input_ids.float().to(self.device)
            batch_attention_mask = batch_attention_mask.float().to(self.device)
            # reconstruction
            outputs, logits_per_time, logits_per_text, total_mask = self.model(batch_x_time, batch_input_ids, batch_attention_mask)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x_time, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name


class CMCInspiredLoss(nn.Module):
    """Opt-in candidate; the existing MindTS training/scoring path is unchanged.

    Inspired by MAD-CMC, IPM 62 (2025) 104013, Eqs. (5)-(16), NOT an exact
    reproduction. Inputs are independent sample/window representations, not
    the four patches/channels of one window relabeled as independent samples.

    Integration contract:
      * Initialize centers from K-means on gradient-training embeddings ONLY.
      * Build a detached P bank and pseudo-label bank on the full training
        population; gather by stable sample IDs for each minibatch.
      * Do not rebuild P from the minibatch, validation set, or test set.
      * Labels here are cluster IDs, never ground-truth anomaly labels.
      * Include this module's center parameters in the training optimizer and
        persist centers, cluster mapping, encoders and preprocessing together.

    Explicit engineering choices: mean rather than summed clustering KL;
    average cross-modal separation over all distinct cluster pairs for K>2;
    skip anchors lacking positive/negative peers and report their count;
    dynamic weighting OFF by default. If enabled, map cosine to [0,1] only
    for the fractional-power weights to avoid NaN on negative cosine values.
    These choices must be reported as adaptations, not attributed to authors.
    """

    def __init__(self, embedding_dim, n_clusters=3, temperature=0.2,
                 inter_weight=0.1, cluster_weight=1.0,
                 dynamic_exponent=None, eps=1e-8):
        super().__init__()
        if not isinstance(embedding_dim, int) or embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer")
        if not isinstance(n_clusters, int) or n_clusters < 2:
            raise ValueError("n_clusters must be an integer >= 2")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if any(not np.isfinite(w) or w < 0 for w in (inter_weight, cluster_weight)):
            raise ValueError("loss weights must be finite and non-negative")
        if not np.isfinite(eps) or not 0 < eps < 0.01:
            raise ValueError("eps must be finite and in (0, 0.01)")
        if dynamic_exponent is not None and (
                not np.isfinite(dynamic_exponent) or dynamic_exponent < 0):
            raise ValueError("dynamic_exponent must be finite and non-negative")
        self.temperature = float(temperature)
        self.inter_weight = float(inter_weight)
        self.cluster_weight = float(cluster_weight)
        self.dynamic_exponent = dynamic_exponent
        self.eps = float(eps)
        self.centers = nn.Parameter(torch.zeros(n_clusters, embedding_dim))
        self.register_buffer("centers_initialized", torch.tensor(False))

    @staticmethod
    def _check_matrix(value, name):
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"{name} must be a 2-D tensor")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite floating-point values")
        if min(value.shape) < 1:
            raise ValueError(f"{name} must not be empty")

    @torch.no_grad()
    def initialize_centers(self, train_kmeans_centers):
        """Caller must fit these centers on the training split, not eval data."""
        self._check_matrix(train_kmeans_centers, "train_kmeans_centers")
        if train_kmeans_centers.shape != self.centers.shape:
            raise ValueError("K-means centers have the wrong shape")
        if torch.unique(train_kmeans_centers, dim=0).shape[0] != self.centers.shape[0]:
            raise ValueError("K-means returned duplicate centers; inspect collapse")
        self.centers.copy_(train_kmeans_centers.to(self.centers))
        self.centers_initialized.fill_(True)

    def soft_assign(self, fused):
        """Student-t assignments, degree of freedom 1 (paper Eq. 13)."""
        self._check_matrix(fused, "fused")
        if not self.centers_initialized.item():
            raise RuntimeError("initialize centers from training K-means first")
        if fused.shape[1] != self.centers.shape[1]:
            raise ValueError("fused embedding dimension differs from centers")
        if fused.device != self.centers.device or fused.dtype != self.centers.dtype:
            raise ValueError("fused embeddings and centers must share device/dtype")
        distances = (fused[:, None, :] - self.centers[None, :, :]).square().sum(-1)
        log_weights = -torch.log1p(distances)
        q = torch.softmax(log_weights, dim=1)
        if not torch.isfinite(q).all():
            raise ValueError("non-finite assignments; inspect embedding scale")
        return q

    @staticmethod
    def make_train_targets(full_train_q, eps=1e-8):
        """Detached DEC target bank (Eq. 14); NEVER call per minibatch/eval split.

        A tensor cannot prove its data provenance. The integration must enforce
        training-only IDs and record them; this helper does not certify that.
        """
        CMCInspiredLoss._check_matrix(full_train_q, "full_train_q")
        if full_train_q.shape[0] < 2 or full_train_q.shape[1] < 2:
            raise ValueError("target bank needs multiple training samples/clusters")
        if not np.isfinite(eps) or not 0 < eps < 0.01:
            raise ValueError("invalid eps")
        q = full_train_q.detach()
        if (q < 0).any() or not torch.allclose(
                q.sum(1), torch.ones_like(q[:, 0]), atol=1e-5, rtol=1e-5):
            raise ValueError("full_train_q rows must be probability distributions")
        weights = q.square() / q.sum(0, keepdim=True).clamp_min(eps)
        return (weights / weights.sum(1, keepdim=True).clamp_min(eps)).detach()

    def _intra(self, features, cluster_ids):
        unit = F.normalize(features, dim=1, eps=self.eps)
        similarity = unit @ unit.T
        peers = ~torch.eye(len(features), dtype=torch.bool, device=features.device)
        same = cluster_ids[:, None] == cluster_ids[None, :]
        positive = same & peers
        negative = ~same & peers
        valid = positive.any(1) & negative.any(1)
        if not valid.any():
            return features.sum() * 0, valid.sum()
        logits = similarity / self.temperature
        if self.dynamic_exponent is not None:
            bounded = ((similarity.detach() + 1) / 2).clamp(self.eps, 1-self.eps)
            base = torch.where(same, 1-bounded, bounded)
            # Stop-gradient weighting is an explicit stability adaptation.
            logits = base.pow(self.dynamic_exponent) * logits
        # Select valid rows before logsumexp: avoid -inf - -inf and NaN gradients.
        numerator = logits[valid].masked_fill(~positive[valid], -torch.inf)
        denominator = logits[valid].masked_fill(~peers[valid], -torch.inf)
        value = (torch.logsumexp(denominator, 1) -
                 torch.logsumexp(numerator, 1)).mean()
        return value, valid.sum()

    def _inter(self, metric_features, log_features, cluster_ids):
        ids = torch.unique(cluster_ids)
        if len(ids) < 2:
            return (metric_features.sum() + log_features.sum()) * 0
        metric_centers = torch.stack([
            metric_features[cluster_ids == key].mean(0) for key in ids])
        log_centers = torch.stack([
            log_features[cluster_ids == key].mean(0) for key in ids])
        metric_centers = F.normalize(metric_centers, dim=1, eps=self.eps)
        log_centers = F.normalize(log_centers, dim=1, eps=self.eps)
        cross = metric_centers @ log_centers.T
        different = ~torch.eye(len(ids), dtype=torch.bool, device=cross.device)
        # Signed cosine, as in Eqs. 9-10: total loss may legitimately be negative.
        return cross[different].mean() + cross.T[different].mean()

    def forward(self, metric_features, log_features, fused, cluster_ids, targets):
        """Return component losses and coverage; do not silently accept B=1."""
        for value, name in ((metric_features, "metric_features"),
                            (log_features, "log_features"), (fused, "fused"),
                            (targets, "targets")):
            self._check_matrix(value, name)
        n = len(fused)
        if n < 2:
            raise ValueError("CMC needs multiple independent samples; B=1 is invalid")
        if metric_features.shape != log_features.shape or len(metric_features) != n:
            raise ValueError("modalities must have matched [B,D] representations")
        if any(v.device != fused.device or v.dtype != fused.dtype
               for v in (metric_features, log_features, targets)):
            raise ValueError("all representations/targets must share device/dtype")
        if (not isinstance(cluster_ids, torch.Tensor) or cluster_ids.shape != (n,)
                or cluster_ids.dtype != torch.long or cluster_ids.device != fused.device):
            raise ValueError("cluster_ids must be a same-device int64 [B] tensor")
        if (cluster_ids < 0).any() or (cluster_ids >= len(self.centers)).any():
            raise ValueError("cluster IDs are outside the configured cluster range")
        if targets.shape != (n, len(self.centers)) or (targets < 0).any():
            raise ValueError("targets must be non-negative [B,K] probabilities")
        if not torch.allclose(targets.sum(1), torch.ones_like(targets[:, 0]),
                              atol=1e-5, rtol=1e-5):
            raise ValueError("target rows must sum to 1")
        q = self.soft_assign(fused)
        target = targets.detach()
        cluster = (target * (target.clamp_min(self.eps).log() -
                            q.clamp_min(self.eps).log())).sum(1).mean()
        intra_metric, valid_anchors = self._intra(metric_features, cluster_ids)
        intra_log, _ = self._intra(log_features, cluster_ids)
        inter = self._inter(metric_features, log_features, cluster_ids)
        total = intra_metric + intra_log + self.inter_weight * inter + self.cluster_weight * cluster
        return {"loss": total, "intra_metric": intra_metric, "intra_log": intra_log,
                "inter": inter, "clustering": cluster, "valid_anchors": valid_anchors,
                "q": q}

    def anomaly_score(self, fused, normal_cluster):
        """Proposed continuous score, NOT the paper's hard fuzzy-cluster rule.

        normal_cluster must be frozen using training-only cluster statistics;
        the largest-cluster-is-normal assumption needs validation in each study.
        This is not wired into MindTS's existing reconstruction score.
        """
        if (not isinstance(normal_cluster, int) or isinstance(normal_cluster, bool)
                or not 0 <= normal_cluster < len(self.centers)):
            raise ValueError("normal_cluster must be a valid integer cluster ID")
        return 1 - self.soft_assign(fused)[:, normal_cluster]
