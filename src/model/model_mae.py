from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.transformer_blocks import Block


class ModelMAE(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        encoder_depth=4,
        decoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        actor_mask_ratio: float = 0.5,
        lane_mask_ratio: float = 0.5,
        history_steps: int = 50,
        future_steps: int = 60,
        loss_weight: List[float] = [1.0, 1.0, 0.35],
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.actor_mask_ratio = actor_mask_ratio
        self.loss_weight = loss_weight

        self.future_embed = AgentEmbeddingLayer(3, 32, drop_path_rate=drop_path)

        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        # decoder
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)

        dpr = [x.item() for x in torch.linspace(0, drop_path, decoder_depth)]
        self.decoder_blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(decoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))

        self.future_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.future_pred = nn.Linear(embed_dim, future_steps * 2)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.future_mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def agent_random_masking(
        fut_tokens, mask_ratio, future_padding_mask, num_actors
    ):
        pred_masks = ~future_padding_mask.all(-1)  # [B, A]
        fut_num_tokens = pred_masks.sum(-1)  # [B]

        len_keeps = (fut_num_tokens * (1 - mask_ratio)).int()
        fut_masked_tokens = []
        fut_keep_ids_list = []
        fut_key_padding_mask = []

        device = fut_tokens.device
        agent_ids = torch.arange(fut_tokens.shape[1], device=device)
        for i, (fut_num_token, len_keep, future_pred_mask) in enumerate(
            zip(fut_num_tokens, len_keeps, pred_masks)
        ):
            pred_agent_ids = agent_ids[future_pred_mask]
            noise = torch.rand(fut_num_token, device=device)
            ids_shuffle = torch.argsort(noise)
            fut_ids_keep = ids_shuffle[:len_keep]
            fut_ids_keep = pred_agent_ids[fut_ids_keep]
            fut_keep_ids_list.append(fut_ids_keep)

            fut_masked_tokens.append(fut_tokens[i, fut_ids_keep])

            fut_key_padding_mask.append(torch.zeros(len_keep, device=device))

        fut_masked_tokens = pad_sequence(fut_masked_tokens, batch_first=True)
        fut_key_padding_mask = pad_sequence(
            fut_key_padding_mask, batch_first=True, padding_value=True
        )

        return (
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        )


    def forward(self, data):
        future_padding_mask = data["padding_mask"][:, :, 50:]
        future_feat = torch.cat([data["y"], ~future_padding_mask[..., None]], dim=-1)
        B, N, L, D = future_feat.shape
        future_feat = future_feat.view(B * N, L, D)
        future_feat = self.future_embed(future_feat.permute(0, 2, 1).contiguous())
        future_feat = future_feat.view(B, N, future_feat.shape[-1])

        # actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]

        # future_feat += actor_type_embed

        (
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        ) = self.agent_random_masking(
            future_feat,
            self.actor_mask_ratio,
            future_padding_mask,
            data["num_actors"],
        )

        x = fut_masked_tokens
        key_padding_mask = fut_key_padding_mask

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # decoding
        x_decoder = self.decoder_embed(x)
        Nf = fut_masked_tokens.shape[1]
        assert x_decoder.shape[1] == Nf
        fut_tokens = x_decoder[:, :Nf]

        decoder_fut_token = self.future_mask_token.repeat(B, N, 1)
        future_pred_mask = ~data["x_key_padding_mask"]
        for i, idx in enumerate(fut_keep_ids_list):
            decoder_fut_token[i, idx] = fut_tokens[i, : len(idx)]
            future_pred_mask[i, idx] = False

        x_decoder = decoder_fut_token

        decoder_key_padding_mask = future_padding_mask.all(-1)

        for blk in self.decoder_blocks:
            x_decoder = blk(x_decoder, key_padding_mask=decoder_key_padding_mask)

        x_decoder = self.decoder_norm(x_decoder)
        future_token = x_decoder.reshape(-1, self.embed_dim)

        # future pred loss
        y_hat = self.future_pred(future_token).view(-1, 60, 2)  # B*N, 120
        y = data["y"].view(-1, 60, 2)
        reg_mask = ~data["x_padding_mask"][:, :, 50:]
        reg_mask[~future_pred_mask] = False
        reg_mask = reg_mask.view(-1, 60)
        future_loss = F.l1_loss(y_hat[reg_mask], y[reg_mask])

        loss = (
            self.loss_weight[0] * future_loss
        )

        out = {
            "loss": loss,
            "future_loss": future_loss.item(),
        }

        if not self.training:
            out["y_hat"] = y_hat.view(1, B, N, 60, 2)
            out["fut_keep_ids"] = fut_keep_ids_list

        return out
