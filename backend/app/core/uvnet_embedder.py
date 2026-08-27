"""UV-Net solid embedding: 128-d L2-normalized geo_vec plus per-face feature histogram."""

from __future__ import annotations

from typing import Any

import dgl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch.conv import NNConv
from dgl.nn.pytorch.glob import MaxPooling
from torch import nn

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover - lightning 2.x package layout
    import lightning.pytorch as pl

from app.config import MODELS_DIR

FACE_LABELS: dict[int, str] = {
    0: "rectangular_through_slot",
    1: "triangular_through_slot",
    2: "rectangular_passage",
    3: "triangular_passage",
    4: "6sides_passage",
    5: "rectangular_through_step",
    6: "2sides_through_step",
    7: "slanted_through_step",
    8: "rectangular_blind_step",
    9: "triangular_blind_step",
    10: "rectangular_blind_slot",
    11: "rectangular_pocket",
    12: "triangular_pocket",
    13: "6sides_pocket",
    14: "chamfer",
    15: "stock",
}

_STOCK_CLASS = 15

_model: pl.LightningModule | None = None


def _conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=False):
    """
    Helper function to create a 1D convolutional layer with batchnorm and LeakyReLU activation

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding size on each side. Defaults to 0.
        bias (bool, optional): Whether bias is used. Defaults to False.

    Returns:
        nn.Sequential: Sequential contained the Conv1d, BatchNorm1d and LeakyReLU layers
    """
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(),
    )


def _conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False):
    """
    Helper function to create a 2D convolutional layer with batchnorm and LeakyReLU activation

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding size on each side. Defaults to 0.
        bias (bool, optional): Whether bias is used. Defaults to False.

    Returns:
        nn.Sequential: Sequential contained the Conv2d, BatchNorm2d and LeakyReLU layers
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )


def _fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
    )


class _MLP(nn.Module):
    """"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        MLP with linear output
        Args:
            num_layers (int): The number of linear layers in the MLP
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden feature dimensions for all hidden layers
            output_dim (int): Output feature dimension

        Raises:
            ValueError: If the given number of layers is <1
        """
        super(_MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # TODO: this could move inside the above loop
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class UVNetCurveEncoder(nn.Module):
    def __init__(self, in_channels=6, output_dims=64):
        """
        This is the 1D convolutional network that extracts features from the B-rep edge
        geometry described as 1D UV-grids (see Section 3.2, Curve & surface convolution
        in paper)

        Args:
            in_channels (int, optional): Number of channels in the edge UV-grids. By default
                                         we expect 3 channels for point coordinates and 3 for
                                         curve tangents. Defaults to 6.
            output_dims (int, optional): Output curve embedding dimension. Defaults to 64.
        """
        super(UVNetCurveEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv1d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = _conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = _conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = _fc(256, output_dims, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class UVNetSurfaceEncoder(nn.Module):
    def __init__(
        self,
        in_channels=7,
        output_dims=64,
    ):
        """
        This is the 2D convolutional network that extracts features from the B-rep face
        geometry described as 2D UV-grids (see Section 3.2, Curve & surface convolution
        in paper)

        Args:
            in_channels (int, optional): Number of channels in the edge UV-grids. By default
                                         we expect 3 channels for point coordinates and 3 for
                                         surface normals and 1 for the trimming mask. Defaults
                                         to 7.
            output_dims (int, optional): Output surface embedding dimension. Defaults to 64.
        """
        super(UVNetSurfaceEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.conv2 = _conv2d(64, 128, 3, padding=1, bias=False)
        self.conv3 = _conv2d(128, 256, 3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(256, output_dims, bias=False)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class _EdgeConv(nn.Module):
    def __init__(
        self,
        edge_feats,
        out_feats,
        node_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        """
        This module implements Eq. 2 from the paper where the edge features are
        updated using the node features at the endpoints.

        Args:
            edge_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(_EdgeConv, self).__init__()
        self.proj = _MLP(1, node_feats, hidden_mlp_dim, edge_feats)
        self.mlp = _MLP(num_mlp_layers, edge_feats, hidden_mlp_dim, out_feats)
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        src, dst = graph.edges()
        proj1, proj2 = self.proj(nfeat[src]), self.proj(nfeat[dst])
        agg = proj1 + proj2
        h = self.mlp((1 + self.eps) * efeat + agg)
        h = F.leaky_relu(self.batchnorm(h))
        return h


class _NodeConv(nn.Module):
    def __init__(
        self,
        node_feats,
        out_feats,
        edge_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        """
        This module implements Eq. 1 from the paper where the node features are
        updated using the neighboring node and edge features.

        Args:
            node_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(_NodeConv, self).__init__()
        self.gconv = NNConv(
            in_feats=node_feats,
            out_feats=out_feats,
            edge_func=nn.Linear(edge_feats, node_feats * out_feats),
            aggregator_type="sum",
            bias=False,
        )
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.mlp = _MLP(num_mlp_layers, node_feats, hidden_mlp_dim, out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        h = (1 + self.eps) * nfeat
        h = self.gconv(graph, h, efeat)
        h = self.mlp(h)
        h = F.leaky_relu(self.batchnorm(h))
        return h


class UVNetGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        input_edge_dim,
        output_dim,
        hidden_dim=64,
        learn_eps=True,
        num_layers=3,
        num_mlp_layers=2,
    ):
        """
        This is the graph neural network used for message-passing features in the
        face-adjacency graph.  (see Section 3.2, Message passing in paper)

        Args:
            input_dim ([type]): [description]
            input_edge_dim ([type]): [description]
            output_dim ([type]): [description]
            hidden_dim (int, optional): [description]. Defaults to 64.
            learn_eps (bool, optional): [description]. Defaults to True.
            num_layers (int, optional): [description]. Defaults to 3.
            num_mlp_layers (int, optional): [description]. Defaults to 2.
        """
        super(UVNetGraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of layers for node and edge feature message passing
        self.node_conv_layers = torch.nn.ModuleList()
        self.edge_conv_layers = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            node_feats = input_dim if layer == 0 else hidden_dim
            edge_feats = input_edge_dim if layer == 0 else hidden_dim
            self.node_conv_layers.append(
                _NodeConv(
                    node_feats=node_feats,
                    out_feats=hidden_dim,
                    edge_feats=edge_feats,
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                ),
            )
            self.edge_conv_layers.append(
                _EdgeConv(
                    edge_feats=edge_feats,
                    out_feats=hidden_dim,
                    node_feats=node_feats,
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                )
            )

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.drop1 = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.5)
        self.pool = MaxPooling()

    def forward(self, g, h, efeat):
        hidden_rep = [h]
        he = efeat

        for i in range(self.num_layers - 1):
            # Update node features
            h = self.node_conv_layers[i](g, h, he)
            # Update edge features
            he = self.edge_conv_layers[i](g, h, he)
            hidden_rep.append(h)

        out = hidden_rep[-1]
        out = self.drop1(out)
        score_over_layer = 0

        # Perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return out, score_over_layer


class _NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        """
        A 3-layer MLP with linear outputs

        Args:
            input_dim (int): Dimension of the input tensor
            num_classes (int): Dimension of the output logits
            dropout (float, optional): Dropout used after each linear layer. Defaults to 0.3.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        """
        Forward pass

        Args:
            inp (torch.tensor): Inputs features to be mapped to logits
                                (batch_size x input_dim)

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class UVNetSegmenter(nn.Module):
    """
    UV-Net solid face segmentation model
    """

    def __init__(
        self,
        num_classes,
        crv_in_channels=6,
        crv_emb_dim=64,
        srf_emb_dim=64,
        graph_emb_dim=128,
        dropout=0.3,
    ):
        """
        Initialize the UV-Net solid face segmentation model

        Args:
            num_classes (int): Number of classes to output per-face
            crv_in_channels (int, optional): Number of input channels for the 1D edge UV-grids
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()
        # A 1D convolutional network to encode B-rep edge geometry represented as 1D UV-grids
        self.curv_encoder = UVNetCurveEncoder(in_channels=crv_in_channels, output_dims=crv_emb_dim)
        # A 2D convolutional network to encode B-rep face geometry represented as 2D UV-grids
        self.surf_encoder = UVNetSurfaceEncoder(in_channels=7, output_dims=srf_emb_dim)
        # A graph neural network that message passes face and edge features
        self.graph_encoder = UVNetGraphEncoder(
            srf_emb_dim,
            crv_emb_dim,
            graph_emb_dim,
        )
        # A non-linear classifier that maps face embeddings to face logits
        self.seg = _NonLinearClassifier(graph_emb_dim + srf_emb_dim, num_classes, dropout=dropout)

    def forward(self, batched_graph):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (total_nodes_in_batch x num_classes)
        """
        # Input features
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        # Compute hidden edge and face features
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        # Message pass and compute per-face(node) and global embeddings
        node_emb, graph_emb = self.graph_encoder(batched_graph, hidden_srf_feat, hidden_crv_feat)
        # Repeat the global graph embedding so that it can be
        # concatenated to the per-node embeddings
        num_nodes_per_graph = batched_graph.batch_num_nodes().to(graph_emb.device)
        graph_emb = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        local_global_feat = torch.cat((node_emb, graph_emb), dim=1)
        # Map to logits
        out = self.seg(local_global_feat)
        return out


class Segmentation(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the segmenter (per-face classifier).
    """

    def __init__(self, num_classes: int, crv_in_channels: int = 6, **kwargs: Any):
        """
        Args:
            num_classes (int): Number of per-face classes in the dataset
            crv_in_channels (int): Number of input channels for the 1D edge UV-grids
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = UVNetSegmenter(num_classes, crv_in_channels=crv_in_channels)

    def forward(self, batched_graph):
        logits = self.model(batched_graph)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


def get_model() -> nn.Module:
    """Lazy-load and cache the UV-Net segmentation model."""
    global _model
    if _model is None:
        ckpt_path = MODELS_DIR / "uvnet_weights.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = ckpt.get("hyper_parameters", {"num_classes": 16, "crv_in_channels": 6})
        model = Segmentation(
            num_classes=hp.get("num_classes", 16),
            crv_in_channels=hp.get("crv_in_channels", 6),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _model = model
    return _model


def embed(graph: dgl.DGLGraph) -> tuple[torch.Tensor, dict[str, int]]:
    """
    Run UV-Net on a preprocessed DGL graph (output of step_to_dgl_graph).

    Returns:
        geo_vec: shape (128,), L2-normalized, dtype float32
        histogram: dict mapping feature label name -> count, excludes "stock" faces
    """
    model = get_model()
    net: UVNetSegmenter = model.model  # type: ignore[assignment]
    with torch.no_grad():
        hidden_crv = net.curv_encoder(graph.edata["x"])
        hidden_srf = net.surf_encoder(graph.ndata["x"])
        node_emb, graph_emb = net.graph_encoder(graph, hidden_srf, hidden_crv)
        n = int(graph.num_nodes())
        graph_emb_rep = graph_emb.repeat_interleave(n, dim=0)
        local_global = torch.cat([node_emb, graph_emb_rep], dim=1)
        logits = net.seg(local_global)

        geo_vec = F.normalize(graph_emb.squeeze(0), dim=0).to(dtype=torch.float32)
        pred_classes = logits.argmax(dim=1).cpu().tolist()

        histogram: dict[str, int] = {}
        for cls_idx in pred_classes:
            if cls_idx == _STOCK_CLASS:
                continue
            label = FACE_LABELS[int(cls_idx)]
            histogram[label] = histogram.get(label, 0) + 1

        if not histogram:
            histogram = {"stock": n}

        return geo_vec, histogram
