import torch
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from ts_benchmark.baselines.MindTS.layers.Embed import WarriorsEmbedding, DataEmbedding_inverted
from ts_benchmark.baselines.MindTS.layers.Transformer_EncDec import Encoder, EncoderLayer
from ts_benchmark.baselines.MindTS.layers.SelfAttention_Family import FullAttention, AttentionLayer
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel


DEEPSEEK_PATH = "Qwen/Qwen2.5-0.5B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, prompt, text_features):
        attn_output, _ = self.attention(prompt, text_features, text_features)
        x = self.norm1(prompt + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        out = self.norm2(x + self.dropout(ff_output))
        
        return out
    

class MultiTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(MultiTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, llm_features, time_features_patch_mask):
        self_attn_output, _ = self.self_attn(llm_features, llm_features, llm_features)
        self_attn_output = self.dropout(self_attn_output)
        cross_attn_output, _ = self.cross_attn(query=time_features_patch_mask, key=self_attn_output, value=self_attn_output)
        cross_attn_output = self.dropout(cross_attn_output)
        x = self.norm1(time_features_patch_mask + cross_attn_output)
        ff_output = self.ffn(x)
        ff_output = self.dropout(ff_output)
        output = self.norm2(x + ff_output)
        
        return output
    

class MINDTSModel(nn.Module):
    def __init__(self, configs):
        super(MINDTSModel, self).__init__()
        self.device = device    # Device (CPU or GPU)
        self.configs = configs    
        self.batch_size = configs.batch_size
        self.seq_len = configs.seq_len 
        self.pred_len = configs.pred_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1    # Number of patches
        self.d_model = configs.d_model
        self.channel_time = configs.enc_in_time    # Number of input time channels
        self.mask_ratio = configs.mask_ratio   # Masking ratio for sequence
        self.llm_layers = 6

        # Load LLM config
        self.deepseek_config = AutoConfig.from_pretrained(DEEPSEEK_PATH,
                                                          trust_remote_code=True)
        self.deepseek_config.num_hidden_layers = self.llm_layers
        self.deepseek_config.output_attentions = False
        self.deepseek_config.output_hidden_states = False
        # MindTS 后面其实不用 attention。output_attentions=True 会让 Qwen2 退回更吃显存的 attention 实现，建议关掉。
        self.llm_hidden_size = self.deepseek_config.hidden_size

        # Embedding layers
        self.windows_data_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.proj_patch = nn.Linear(self.patch_num * configs.d_model, configs.seq_len, bias=True)
        self.prompt_proj_hidden = nn.Linear(self.llm_hidden_size, configs.d_model, bias=True)
        self.text_proj_hidden = nn.Linear(self.llm_hidden_size, configs.d_model, bias=True)
        self.proj_text = nn.Linear(256 * configs.d_model, configs.d_model, bias=True)
        self.proj_prompt = nn.Linear(128 * configs.d_model, configs.d_model, bias=True)
        self.patch_embedding = WarriorsEmbedding(configs.d_model, self.patch_size, self.stride, self.stride, configs.dropout)

        # Time patch encoder with stacked attention layers
        self.time_patch_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )
        self.time_windows_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )
        self.layer = configs.e_layers    # Number of encoder layers
        self.layer_norm = nn.LayerNorm(configs.d_model)    # Layer normalization
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))    # Logit scaling factor
        self.top_k = self.patch_size    # Top-k value for selection
        self.d_ff = configs.d_ff    # Feedforward hidden dimension
        self.num_heads = configs.n_heads    # Number of attention heads

        self.tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_PATH, trust_remote_code=True)
        #self.model = AutoModelForCausalLM.from_pretrained(DEEPSEEK_PATH, trust_remote_code=True, config=self.deepseek_config)
        self.model = AutoModel.from_pretrained(
            DEEPSEEK_PATH,
            trust_remote_code=True,
            config=self.deepseek_config,
            torch_dtype=torch.float16,
            )
        for param in self.model.parameters():
            param.requires_grad = False
        self.transformer_block = TransformerBlock(self.d_model, self.num_heads, self.d_ff)
        self.multimodal_Transformer_Block = MultiTransformerBlock(self.d_model, self.num_heads, self.d_ff)
        self.prob_net = nn.Sequential(nn.PReLU(), nn.Linear(configs.d_model, 1), nn.Sigmoid())

    def random_masking(self, xb, mask_ratio):
        bs_nvars, L, d_model = xb.shape
        x = xb.clone()
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(bs_nvars, L, device=self.device)
        ids_shuffle = torch.argsort(noise, dim=1).to(self.device)
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(self.device)
        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d_model))
        x_removed = torch.zeros(bs_nvars, L - len_keep, d_model, device=self.device)
        x_ = torch.cat([x_kept, x_removed], dim=1)
        x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, d_model))
        mask = torch.ones([bs_nvars, L], device=self.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
    
        return x_masked, x_kept, mask, ids_restore

    def calcute_lags(self, x_enc): 
        q_fft = torch.fft.rfft(x_enc.contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        _, lags = torch.topk(corr, self.top_k, dim=-1)
        return lags
    
    
    def Multimodal_Time_Series(self, x_enc_time, x_enc_input_ids, x_enc_attention_mask):
        # -------------------------------------------------------------Input data normalization--------------------------------------------------------------------
        means = x_enc_time.mean(1, keepdim=True).detach()
        x_enc_time = x_enc_time - means
        stdev = torch.sqrt(torch.var(x_enc_time, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_time /= stdev
        B, T, N = x_enc_time.size()

        # -------------------------------------------------------------Series patching and masking-----------------------------------------------------------------
        x_enc_time_patch_normal, _ = self.patch_embedding(x_enc_time.permute(0, 2, 1))
        x_enc_time_patch_mask, _, _, _ = self.random_masking(x_enc_time_patch_normal, self.mask_ratio)

        # -------------------------------------------------------------Time Encoder--------------------------------------------------------------------------------
        time_features_patch_normal, attns = self.time_patch_encoder(x_enc_time_patch_normal)    #[B*C, N, D]
        time_features_patch_mask, attns = self.time_patch_encoder(x_enc_time_patch_mask)    #[B*C, N, D]

        # -------------------------------------------------------------prompt Generation --------------------------------------------------------------------------
        x_enc_time = x_enc_time.permute(0, 2, 1).contiguous()
        x_enc_time = rearrange(x_enc_time, 'b c l -> (b c) l')
        x_enc_time = x_enc_time.unfold(1, self.patch_size, self.stride) # (B * N, Patch_num, Patch_size)

        min_values = torch.min(x_enc_time, dim=2)[0]
        max_values = torch.max(x_enc_time, dim=2)[0]
        medians = torch.median(x_enc_time, dim=2).values
        lags = self.calcute_lags(x_enc_time)
        trends = x_enc_time.diff(dim=2)
        self.description = 'MDT datasets include numerical stock data from Yahoo Finance and news information collected from various financial news websites such as NASDAQ, Bloomberg, and others.'
        prompt_list = []
        for b in range(x_enc_time.shape[0]):
            prompt = []
            for c in range(x_enc_time.shape[1]):
                min_values_str = str(min_values[b][c].tolist())
                max_values_str = str(max_values[b][c].tolist())
                median_values_str = str(medians[b][c].tolist())
                lags_values_str = str(lags[b][c].tolist())
                patch_num_middle = self.patch_num // 2
                first_half = trends[b][:patch_num_middle]
                second_half = trends[b][patch_num_middle:]
                first_half_mean = first_half.mean()
                second_half_mean = second_half.mean()
                first_half_std = first_half.std()
                second_half_std = second_half.std()
                if first_half_mean > 0 and second_half_mean < 0:
                    trend = 'first upward then downward'
                elif first_half_mean < 0 and second_half_mean > 0:
                    trend = 'first downward then upward'
                elif first_half_mean > 0 and second_half_mean > 0:
                    trend = 'upwarding'
                elif first_half_mean < 0 and second_half_mean < 0:
                    trend = 'downwarding'
                elif first_half_std < 0.01 and second_half_std < 0.01:
                    trend = 'balanced'
                else:
                    trend = 'uncertain'

                prompt_ = (
                    f"<|start_prompt|>Dataset description: {self.description}"
                    f"Task description: reconstruct the {str(self.seq_len)} steps given the previous {str(self.seq_len)} steps information; "
                    "Input statistics: "
                    f"min value {min_values_str}, "
                    f"max value {max_values_str}, "
                    f"median value {median_values_str}, "
                    f"the trend of input is {trend}, "
                    f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
                )
                prompt.append(prompt_)
            prompt_list.append(prompt)
        
        # -------------------------------------------------------------prompt Reasoning---------------------------------------------------------------------------
        all_prompts = [prompt for batch in prompt_list for prompt in batch]
        prompt_tokens = self.tokenizer(all_prompts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = prompt_tokens["input_ids"].to(self.device)
        attention_mask = prompt_tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
                )
            embeddings = outputs.last_hidden_state
            embeddings = embeddings.detach() 

        del input_ids, attention_mask, outputs

        total_prompts = len(all_prompts)
        batch_size_prompt = len(prompt_list[0])
        all_embeddings = embeddings.view(-1, batch_size_prompt, embeddings.size(1), embeddings.size(2))
        all_embeddings = all_embeddings.squeeze(2)

        prompt_feature = all_embeddings.to(torch.float32)
        del embeddings, all_embeddings

        prompt_feature = self.prompt_proj_hidden(prompt_feature)
        prompt_feature = rearrange(prompt_feature, 'v n m d -> v n (m d)', n = self.patch_num, m = 128, d = self.d_model)
        prompt_feature = self.proj_prompt(prompt_feature)      

        # -------------------------------------------------------------text Reasoning------------------------------------------------------------------------------
        with torch.no_grad():
            outputs = self.model(
                input_ids=x_enc_input_ids.long(),
                attention_mask=x_enc_attention_mask.long(),
                return_dict=True
            )
            embeddings = outputs.last_hidden_state
            embeddings = embeddings.detach()

        text_features = embeddings.to(torch.float32)

        text_features = self.prompt_proj_hidden(text_features)
        text_features = rearrange(text_features, 'b m h -> b (m h)', m = 256, h = self.d_model)
        text_features = text_features.unsqueeze(1)
        text_features = self.proj_text(text_features)
        text_features = text_features.repeat(self.channel_time, 1, 1) 

        # -------------------------------------------------------------prompt and textCross-view Attention--------------------------------------------------------
        llm_features = self.transformer_block(prompt_feature, text_features)

        # -------------------------------------------------------------time-text Similarity matrix----------------------------------------------------------------
        time_norm = F.normalize(time_features_patch_normal, p=2, dim=-1)
        llm_norm = F.normalize(llm_features, p=2, dim=-1)
        logit_scale = self.logit_scale.exp()
        logits_per_time = logit_scale * torch.bmm(time_norm, llm_norm.transpose(1, 2))
        logits_per_text = logits_per_time.transpose(1, 2)

        # -------------------------------------------------------------Information Condenser----------------------------------------------------------------------
        total_mask = self.prob_net(llm_features)
        if total_mask.shape[-1] == 1:
            inv_probs = 1 - total_mask
            total_mask_prob = torch.cat([inv_probs, total_mask], dim=-1)
        else:
            total_mask_prob = total_mask.softmax(dim=-1)
        total_mask_reparameterize = torch.nn.functional.gumbel_softmax(torch.log(total_mask_prob + 1e-6), tau = 1, hard = True)[...,1]
        total_mask_reparameterize = total_mask_reparameterize.unsqueeze(-1)
        llm_features = total_mask_reparameterize * llm_features      

        # -------------------------------------------------------------Reconstruction-----------------------------------------------------------------------------
        multi_features = self.multimodal_Transformer_Block(llm_features, time_features_patch_mask)
        output = rearrange(multi_features, '(b c) n d -> (b c) (n d)', c = self.channel_time, n = self.patch_num, d = self.d_model)
        output = self.proj_patch(output)
        output = rearrange(output, '(b c) t -> b t c', t = self.seq_len, c = self.channel_time)

        # -------------------------------------------------------------Inverse normalization-----------------------------------------------------------------------
        output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return output, logits_per_time, logits_per_text, total_mask

    
    def forward(self, x_enc_time, x_enc_input_ids, x_enc_attention_mask):
        outputs, logits_per_time, logits_per_text, total_mask = self.Multimodal_Time_Series(x_enc_time, x_enc_input_ids, x_enc_attention_mask)
        return outputs, logits_per_time, logits_per_text, total_mask
