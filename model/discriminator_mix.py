import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

try:
    from transformers import HubertModel
except ImportError:
    HubertModel = None

class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Adapter, self).__init__()
        self.adapter = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.LeakyReLU(negative_slope=0.01), #nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim))  # Ensure output matches HuBERT hidden size
        )

    def forward(self, x):
        return self.adapter(x)

class MultiHeadAttentivePooling(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        
    def forward(self, x):
        query = self.query.expand(x.size(0), -1, -1)
        attn_output, _ = self.attention(query, x, x)
        return attn_output.squeeze(1)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = spectral_norm(nn.Linear(in_features, out_features))
        self.linear2 = spectral_norm(nn.Linear(out_features, out_features))
        self.shortcut = spectral_norm(nn.Linear(in_features, out_features)) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.leaky_relu(self.linear1(x), 0.2)
        out = self.linear2(out)
        return F.leaky_relu(out + identity, 0.2)
class EnhancedAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super(EnhancedAdapter, self).__init__()
        
        # First projection with intermediate size
        self.down_project = spectral_norm(nn.Linear(input_dim, hidden_dim))
        
        # Add Layer Normalization for better training stability
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Intermediate processing
        self.intermediate = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim * 2)),
            nn.GELU(),  # GELU typically works better than LeakyReLU for transformers
            nn.Dropout(dropout_rate),
            spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim))
        )
        
        # Skip connection to help with gradient flow
        self.skip_connection = spectral_norm(nn.Linear(input_dim, hidden_dim)) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        # Skip connection path
        identity = self.skip_connection(x)
        
        # Main path
        out = self.down_project(x)
        out = self.norm1(out)
        
        # Intermediate processing
        out = self.intermediate(out)
        out = self.norm2(out)
        
        # Add skip connection
        return out + identity
class SpeakerAttentionPool(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.speaker_queries = nn.Parameter(torch.randn(1, 3, hidden_size))
        self.query_weights = nn.Parameter(torch.ones(3))
        
    def forward(self, x):
        queries = self.speaker_queries.expand(x.size(0), -1, -1)
        attn_outputs = []        
        for i in range(3):
            query = queries[:, i:i+1]
            attn_output, _ = self.attention(query, x, x)
            attn_outputs.append(attn_output * F.softmax(self.query_weights, dim=0)[i])        
        combined = torch.sum(torch.stack(attn_outputs, dim=1), dim=1)
        return combined.squeeze(1)

class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.linear1 = spectral_norm(nn.Linear(in_features, out_features))
        self.norm2 = nn.LayerNorm(out_features)
        self.linear2 = spectral_norm(nn.Linear(out_features, out_features))
        self.dropout = nn.Dropout(0.1)
        self.shortcut = spectral_norm(nn.Linear(in_features, out_features)) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.norm1(x)
        out = F.gelu(self.linear1(out))
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.linear2(out)
        
        return F.gelu(out + identity)


class MinibatchStdDev(nn.Module):
    """
    Adds one feature containing the average feature-wise std across the batch.
    Helps discriminator detect distribution-level differences.
    """
    def forward(self, x):
        if x.size(0) <= 1:
            std_feat = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        else:
            std = x.float().std(dim=0, unbiased=False).mean()
            std_feat = std.expand(x.size(0), 1).to(dtype=x.dtype)

        return torch.cat([x, std_feat], dim=1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super().__init__()

        self.main = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, out_dim)),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            spectral_norm(nn.Linear(out_dim, out_dim)),
            nn.LayerNorm(out_dim),
        )

        self.skip = (
            spectral_norm(nn.Linear(in_dim, out_dim))
            if in_dim != out_dim
            else nn.Identity()
        )

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))


class SimpleDiscriminator(nn.Module):
    """Small embedding discriminator with two residual MLP blocks."""

    def __init__(self, emb_dim=192, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        self.discriminator = nn.Sequential(
            ResidualMLPBlock(emb_dim, hidden_dim, dropout_rate),
            ResidualMLPBlock(hidden_dim, hidden_dim, dropout_rate),
            spectral_norm(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, x):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(
                "SimpleDiscriminator expects pooled embeddings of shape (B, E), "
                f"but got {tuple(x.shape)}"
            )
        return self.discriminator(F.normalize(x, p=2, dim=-1))


class InterDiscrim(nn.Module):
    """Intermediate embedding discriminator with wider residual MLP capacity."""

    def __init__(
        self,
        emb_dim=192,
        hidden_dim=512,
        mid_dim=384,
        head_dim=128,
        dropout_rate=0.15,
    ):
        super().__init__()
        self.discriminator = nn.Sequential(
            ResidualMLPBlock(emb_dim, hidden_dim, dropout_rate),
            ResidualMLPBlock(hidden_dim, hidden_dim, dropout_rate),
            ResidualMLPBlock(hidden_dim, mid_dim, dropout_rate),
            ResidualMLPBlock(mid_dim, mid_dim, dropout_rate),
            MinibatchStdDev(),
            spectral_norm(nn.Linear(mid_dim + 1, head_dim)),
            nn.LayerNorm(head_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            spectral_norm(nn.Linear(head_dim, 1)),
        )

    def forward(self, x):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(
                "InterDiscrim expects pooled embeddings of shape (B, E), "
                f"but got {tuple(x.shape)}"
            )
        return self.discriminator(F.normalize(x, p=2, dim=-1))


class Discriminator2(nn.Module):
    def __init__(
        self,
        emb_dim=192,
        hidden_dim=384,
        mid_dim=256,
        dropout_rate=0.1,
        hubert_model_name=None,
        cache_dir=None,
        proj_dim=None,
    ):
        super().__init__()

        self.discriminator = nn.Sequential(
            ResidualMLPBlock(emb_dim, hidden_dim, dropout_rate),
            ResidualMLPBlock(hidden_dim, hidden_dim, dropout_rate),
            ResidualMLPBlock(hidden_dim, mid_dim, dropout_rate),
            MinibatchStdDev(),
            spectral_norm(nn.Linear(mid_dim + 1, 128)),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            spectral_norm(nn.Linear(128, 1)),
        )

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        return self.discriminator(x)


class MixupDiscriminator(nn.Module):
    def __init__(self, hubert_model_name="facebook/hubert-large-ls960-ft", cache_dir="", proj_dim=256, emb_dim=192):
        super(MixupDiscriminator, self).__init__()
        if HubertModel is None:
            raise ImportError("transformers is required to use MixupDiscriminator")
        self.hubert = HubertModel.from_pretrained(hubert_model_name, cache_dir=cache_dir)
        
        # For speaker recognition, layers 7-12 are most informative for speaker characteristics
        hidden_size = self.hubert.config.hidden_size
        self.projection_7 = spectral_norm(nn.Linear(hidden_size, proj_dim))
        self.projection_9 = spectral_norm(nn.Linear(hidden_size, proj_dim))
        self.projection_11 = spectral_norm(nn.Linear(hidden_size, proj_dim))
        self.projection_12 = spectral_norm(nn.Linear(hidden_size, proj_dim))
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Enhanced attention pooling with multi-head attention
        self.attn_pool = MultiHeadAttentivePooling(hidden_size, num_heads=8)
        
        # Weighted layer combination
        self.layer_weights = nn.Parameter(torch.ones(4))
        
        # Enhanced discriminator with residual connections
        self.discriminator = nn.Sequential(
            ResidualBlock(proj_dim * 4, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            ResidualBlock(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(256, 1))
        )
        
        # Improved adapter with skip connection
        self.adapter = EnhancedAdapter(input_dim=emb_dim, hidden_dim=hidden_size)
        
    def forward(self, input_audio):
        adapted_embeddings = self.adapter(input_audio)
        if adapted_embeddings.dim() == 2:
            adapted_embeddings = adapted_embeddings.unsqueeze(1)
            
        encoder_outputs = self.hubert.encoder(
            hidden_states=adapted_embeddings,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = encoder_outputs.hidden_states
        
        # Use higher layers (7, 9, 11, 12) which are better for speaker characteristics
        layer_projections = []
        for idx, (layer_idx, projection) in enumerate([
            (7, self.projection_7),
            (9, self.projection_9),
            (11, self.projection_11),
            (12, self.projection_12)
        ]):
            # Apply layer norm before projection
            normalized = self.layer_norm(hidden_states[layer_idx])
            # Apply attention pooling
            pooled = self.attn_pool(normalized)
            # Project with learned weight
            projected = projection(pooled) * F.softmax(self.layer_weights, dim=0)[idx]
            layer_projections.append(projected)
            
        # Weighted concatenation of layer projections
        concat_proj = torch.cat(layer_projections, dim=-1)
        
        # Final discrimination
        return self.discriminator(concat_proj)


class HubertDiscriminator(nn.Module):
    def __init__(self, hubert_model_name="facebook/hubert-large-ls960-ft",cache_dir="", proj_dim=128, emb_dim = 192):
        super(HubertDiscriminator, self).__init__()
        if HubertModel is None:
            raise ImportError("transformers is required to use HubertDiscriminator")
        # Load pre-trained HuBERT
        self.hubert = HubertModel.from_pretrained(hubert_model_name, cache_dir=cache_dir)
        # for param in self.hubert.parameters():
        #     param.requires_grad = False  # Freeze HuBERT

        # Define projection layers for layers 3, 6, 9, 12
        hidden_size = self.hubert.config.hidden_size
        self.projection_3 = spectral_norm(nn.Linear(hidden_size, proj_dim))
        self.projection_6 = spectral_norm(nn.Linear(hidden_size, proj_dim))
        self.projection_9 = spectral_norm(nn.Linear(hidden_size, proj_dim))
        self.projection_12 = spectral_norm(nn.Linear(hidden_size, proj_dim))
        
        # Discriminator head
        self.discriminator = nn.Sequential(
            spectral_norm(nn.Linear(proj_dim * 4, 256)),  # Concatenate 4 layers' projections
            nn.LeakyReLU(negative_slope=0.01), #nn.ReLU(),
            spectral_norm(nn.Linear(256, 1)),  # Binary classification for real/fake
            #nn.Sigmoid()
        )
        # Adapter to align speaker encoder outputs with HuBERT
        self.adapter = Adapter(input_dim=emb_dim, hidden_dim=hidden_size)
    
    def forward(self, input_audio):
        # Extract HuBERT hidden states
        # Use adapter to align embeddings with HuBERT hidden size
        adapted_embeddings = self.adapter(input_audio)  # (batch_size, hidden_size)
         # Reshape embeddings to include a sequence length (if needed)
        if adapted_embeddings.dim() == 2:  # (batch_size, hidden_size)
            adapted_embeddings = adapted_embeddings.unsqueeze(1)  # Add sequence length: (batch_size, seq_len=1, hidden_size)
        # with torch.no_grad():  # Ensure HuBERT remains frozen
        #     outputs = self.hubert(input_audio, output_hidden_states=True)
        #     hidden_states = outputs.hidden_states
        # with torch.no_grad():  # Ensure HuBERT remains frozen
            # Feed embeddings into the HuBERT transformer (bypass feature extractor)
        encoder_outputs = self.hubert.encoder(
            hidden_states=adapted_embeddings,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = encoder_outputs.hidden_states
        # Select layers 3, 6, 9, 12
        layer_3 = hidden_states[3]
        layer_6 = hidden_states[6]
        layer_9 = hidden_states[9]
        layer_12 = hidden_states[12]
        
        # Apply projection layers
        proj_3 = self.projection_3(layer_3.mean(dim=1))  # Pool along time if necessary
        proj_6 = self.projection_6(layer_6.mean(dim=1))
        proj_9 = self.projection_9(layer_9.mean(dim=1))
        proj_12 = self.projection_12(layer_12.mean(dim=1))
        
        # Concatenate projections
        concat_proj = torch.cat([proj_3, proj_6, proj_9, proj_12], dim=-1)
        
        # Pass through discriminator head
        real_fake_logits = self.discriminator(concat_proj)
        return real_fake_logits


class Discriminator_spectral(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = spectral_norm(nn.Linear(embedding_dim, 128))
        self.activation = nn.LeakyReLU(0.2)
        self.fc2 = spectral_norm(nn.Linear(128, 1))

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)
        #return torch.sigmoid(self.fc2(x))

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(Discriminator, self).__init__()
        # self.conv1 = nn.Conv1d(1, d, 4, 2, 1)
        # self.conv2 = nn.Conv1d(d, d * 2, 4, 2, 1)
        # self.conv2_bn = nn.BatchNorm2d(d * 2)
        # self.conv3 = nn.Conv1d(d * 2, d * 4, 4, 2, 1)
        # self.conv3_bn = nn.BatchNorm2d(d * 4)
        # self.conv4 = nn.Conv1d(d * 4, d * 8, 4, 1, 1)
        # self.conv4_bn = nn.BatchNorm2d(d * 8)
        # self.conv5 = nn.Conv1d(d * 8, 1, 4, 1, 1)
        self.conv1 = spectral_norm(nn.Conv1d(1, d, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv1d(d, d * 2, 4, 2, 1))
        self.conv2_bn = nn.BatchNorm1d(d * 2)  # Changed to BatchNorm1d for 1D Conv
        self.conv3 = spectral_norm(nn.Conv1d(d * 2, d * 4, 4, 2, 1))
        self.conv3_bn = nn.BatchNorm1d(d * 4)  # Changed to BatchNorm1d for 1D Conv
        self.conv4 = spectral_norm(nn.Conv1d(d * 4, d * 8, 4, 1, 1))
        self.conv4_bn = nn.BatchNorm1d(d * 8)  # Changed to BatchNorm1d for 1D Conv
        self.conv5 = spectral_norm(nn.Conv1d(d * 8, 1, 4, 1, 1))

    def forward(self, input):
        #x = torch.cat([input, label], 1)
        x = input.unsqueeze(1)  #
        # x = input
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        #x = F.sigmoid(self.conv5(x))
        x = x.squeeze(2)
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
