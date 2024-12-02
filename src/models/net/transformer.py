import torch
import torch.nn as nn

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.net.embed import PositionalEncoding, TimeEmbedding

class Transformer1D(nn.Module):
    def __init__(self, 
                d_input: int,
                d_cond: int | None,
                d_model: int,
                month_embedder: TimeEmbedding,
                day_embedder: TimeEmbedding,
                pos_encoder: PositionalEncoding,
                input_encoder: nn.TransformerEncoder,
                cond_encoder: nn.TransformerEncoder | None,
                decoder: nn.TransformerDecoder,
                augment: bool = False,
                dropout: float = 0.1) -> None:
        super(Transformer1D, self).__init__()

        self.augment = augment
        if self.augment:
            self.dropout = nn.Dropout(p=dropout)

        self.input_linear = nn.Linear(in_features=d_input, out_features=d_model)

        if d_cond is not None:
            self.cond_linear = nn.Linear(in_features=d_cond, out_features=d_model)

        self.day_embedder = day_embedder
        self.month_embedder = month_embedder
        self.pos_encoder = pos_encoder 
        self.input_encoder = input_encoder
        self.cond_encoder = cond_encoder
        self.decoder = decoder

        self.output_linear = nn.Linear(in_features=d_model, out_features=d_input)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):

        input_month = x[:,:,-2].to(torch.int64)
        input_day = x[:,:,-1].to(torch.int64)
        x = self.dropout(x[:,:,:-2]) if self.augment else x[:,:,:-2]
        batch, input_length, _ = x.shape
        x = x.view(batch * input_length, -1)
        x = self.input_linear(x)
        x = x.view(batch, input_length, -1)
        x += self.month_embedder(input_month) + self.day_embedder(input_day)
        x = self.pos_encoder(x)
        input_encode = self.input_encoder(x)

        cond_encode = None
        if cond is not None:
            cond_month = cond[:,:,-2].to(torch.int64)
            cond_day = cond[:,:,-1].to(torch.int64)
            cond = self.dropout(cond[:,:,:-2]) if self.augment else cond[:,:,:-2]
            batch, cond_length, _ = cond.shape
            cond = cond.view(batch * cond_length, -1)
            cond = self.cond_linear(cond)
            cond = cond.view(batch, cond_length, -1)
            cond = self.month_embedder(cond_month) + self.day_embedder(cond_day)
            cond = self.pos_encoder(cond)
            cond_encode = self.cond_encoder(cond)

        out = self.decoder(tgt=self.pos_encoder(input_encode),
                            memory=input_encode if cond_encode is None else cond_encode,
                            tgt_mask=self.get_tgt_mask(input_encode.shape[1]))

        out = out.view(batch * input_length, -1)
        out = self.output_linear(out)
        out = out.view(batch, input_length, -1)

        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

if __name__ == "__main__":
    transformer = Transformer1D(d_input=12, 
                                d_cond=5, 
                                d_model=32,
                                month_embedder=TimeEmbedding(d_model=32,
                                                            dropout=0.1,
                                                            max_len=15,
                                                            frequency=5000.0,
                                                            amplitude=2.0),
                                day_embedder=TimeEmbedding(d_model=32,
                                                        dropout=0.1,
                                                        max_len=1500,
                                                        frequency=10000.0,
                                                        amplitude=1.0),
                                pos_encoder=PositionalEncoding(d_model=32, 
                                                                dropout=0.1, 
                                                                max_len=40,
                                                                frequency=1000.0),
                                input_encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=32,
                                                                                                nhead=1,
                                                                                                dim_feedforward=64,
                                                                                                dropout=0.1,
                                                                                                batch_first=True),
                                                                                                num_layers=2),
                                cond_encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=32,
                                                                                                nhead=1,
                                                                                                dim_feedforward=64,
                                                                                                dropout=0.1,
                                                                                                batch_first=True),
                                                                                                num_layers=2),
                                decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=32, 
                                                                                        nhead=1,
                                                                                        dim_feedforward=64,
                                                                                        dropout=0.1,
                                                                                        batch_first=True),
                                                                                        num_layers=2),
                                augment=True, 
                                dropout=0.1)

    input = torch.randn(10, 29, 12)
    cond = torch.randn(10, 36, 5)

    input = torch.cat((input, torch.randint(0, 12, (10, 29, 1)), torch.randint(0, 100, (10, 29, 1))), dim = 2)
    cond = torch.cat((cond, torch.randint(0, 12, (10, 36, 1)), torch.randint(0, 100, (10, 36, 1))), dim = 2)

    out = transformer(input, cond)
    print(out.shape)