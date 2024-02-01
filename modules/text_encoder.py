import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, encode_dim, pad_id):
        super().__init__(
            num_embeddings=vocab_size, embedding_dim=encode_dim, padding_idx=pad_id,
        )


class PositionEmbedding(nn.Embedding):
    def __init__(self, max_len, encode_dim):
        super().__init__(num_embeddings=max_len, embedding_dim=encode_dim)


# class SegmentEmbedding(nn.Embedding):
#     def __init__(self, encode_dim):
#         super().__init__(num_embeddings=2, embedding_dim=encode_dim)


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, pad_id, encode_dim, drop_prob=0.1):
        super().__init__()

        self.token_embed = TokenEmbedding(
            vocab_size=vocab_size, encode_dim=encode_dim, pad_id=pad_id,
        )
        self.pos_embed = PositionEmbedding(max_len=max_len, encode_dim=encode_dim)
        # self.seg_embed = SegmentEmbedding(encode_dim)

        self.pos = torch.arange(max_len, dtype=torch.long).unsqueeze(0)

        self.norm = nn.LayerNorm(encode_dim)
        self.embed_drop = nn.Dropout(drop_prob)

    # def forward(self, token_id, seg_ids):
    def forward(self, token_id):
        b, seq_len = token_id.shape

        x = self.token_embed(token_id)
        x += self.pos_embed(self.pos[:, : seq_len].repeat(b, 1).to(token_id.device))
        # x += self.seg_embed(seg_ids)

        x = self.norm(x)
        x = self.embed_drop(x)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, encode_dim, drop_prob=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(encode_dim)
        self.resid_drop = nn.Dropout(drop_prob)

    def forward(self, x, sub_layer):
        skip = x.clone()
        x = self.norm(x)
        x = sub_layer(x)
        x += skip
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, encode_dim, n_heads, drop_prob=0.1):
        super().__init__()
    
        self.n_heads = n_heads

        self.head_dim = encode_dim // n_heads

        self.qkv_proj = nn.Linear(encode_dim, 3 * n_heads * self.head_dim, bias=False)
        self.attn_drop = nn.Dropout(drop_prob)
        self.out_proj = nn.Linear(encode_dim, encode_dim, bias=False)

    @staticmethod
    def _get_attention_score(q, k):
        attn_score = torch.einsum("bnid,bnjd->bnij", q, k)
        return attn_score

    def forward(self, x, mask=None):
        q, k, v = torch.split(
            self.qkv_proj(x), split_size_or_sections=self.n_heads * self.head_dim, dim=2,
        )
        q = rearrange(q, pattern="b i (n d) -> b n i d", h=self.n_heads, d=self.head_dim)
        k = rearrange(k, pattern="b i (n d) -> b n i d", h=self.n_heads, d=self.head_dim)
        v = rearrange(v, pattern="b i (n d) -> b n i d", h=self.n_heads, d=self.head_dim)

        attn_score = self._get_attention_score(q=q, k=k)
        if mask is not None:
            attn_score.masked_fill_(mask=mask, value=-1e9)
            attn_score /= (self.head_dim ** 0.5)
        attn_weight = F.softmax(attn_score, dim=3)

        x = torch.einsum("bhnm,bhmd->bhnd", attn_weight, v)
        x = rearrange(x, pattern="b n i d -> b i (n d)")
        x = self.attn_drop(x)

        x = self.out_proj(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, encode_dim, mlp_size, drop_prob=0.1):
        super().__init__()

        self.proj1 = nn.Linear(encode_dim, mlp_size)
        self.proj2 = nn.Linear(mlp_size, encode_dim)
        self.mlp_drop2 = nn.Dropout(drop_prob)
        self.mlp_drop1 = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.proj1(x)
        x = F.gelu(x)
        x = self.mlp_drop1(x)
        x = self.proj2(x)
        x = self.mlp_drop2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, encode_dim, n_heads, mlp_size, drop_prob=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            encode_dim=encode_dim, n_heads=n_heads, drop_prob=drop_prob,
        )
        self.attn_resid_conn = ResidualConnection(
            encode_dim=encode_dim, drop_prob=drop_prob,
        )
        self.feed_forward = PositionwiseFeedForward(
            encode_dim=encode_dim, mlp_size=mlp_size,
        )
        self.ff_resid_conn = ResidualConnection(
            encode_dim=encode_dim, drop_prob=drop_prob,
        )

    def forward(self, x, mask=None):
        x = self.attn_resid_conn(x=x, sub_layer=lambda x: self.self_attn(x, mask=mask))
        x = self.ff_resid_conn(x=x, sub_layer=self.feed_forward)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self, depth, n_heads, encode_dim, mlp_size, drop_prob
    ):
        super().__init__()

        self.enc_stack = nn.ModuleList([
            TransformerLayer(
                n_heads=n_heads,
                encode_dim=encode_dim,
                mlp_size=mlp_size,
                drop_prob=drop_prob,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(encode_dim)

    def forward(self, x, mask):
        for enc_layer in self.enc_stack:
            x = enc_layer(x, mask=mask)
        x = self.norm(x)
        return x


# class TextEncoder(nn.Module):
class TextConditioner(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        max_len=77, # "$M$"
        depth=32, # "$N$"
        n_heads=12,
        encode_dim=1280, # "$d_{\tau}$"
        mlp_size=768 * 4,
        drop_prob=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.encode_dim = encode_dim
        self.pad_id = pad_id

        self.embed = BERTEmbedding(
            vocab_size=vocab_size,
            max_len=max_len,
            pad_id=pad_id,
            encode_dim=encode_dim,
            drop_prob=drop_prob,
        )
        self.tf_block = TransformerBlock(
            depth=depth,
            n_heads=n_heads,
            encode_dim=encode_dim,
            mlp_size=mlp_size,
            drop_prob=drop_prob,
        )

    def _get_pad_mask(self, token_id):
        mask = (token_id == self.pad_id).unsqueeze(1).unsqueeze(2)
        mask.requires_grad = False
        return mask

    def forward(self, token_id, seg_ids):
        x = self.embed(token_id=token_id, seg_ids=seg_ids)
        pad_mask = self._get_pad_mask(token_id)
        x = self.tf_block(x, mask=pad_mask)
        return x

