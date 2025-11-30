from neuralop.models import FNO

operator = FNO(n_modes=(32, 32),
               hidden_channels=64,
               in_channels=2,
               out_channels=1)