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

        self.valid_data_loader = anomaly_detection_multi_data_provider(
            valid_data,
            valid_text,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
        )

        self.train_data_loader = anomaly_detection_multi_data_provider(
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

        self.thre_loader = anomaly_detection_multi_data_provider(
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

        self.test_data_loader = anomaly_detection_multi_data_provider(
            test_data,
            test_text,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="test",
        )

        self.thre_loader = anomaly_detection_multi_data_provider(
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
