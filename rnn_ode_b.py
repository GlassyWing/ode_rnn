import torch.nn as nn
import torch
from torch.optim import Adam, SGD
import numpy as np
import matplotlib.pyplot as plot


class ODE_RNN(nn.Module):

    def __init__(self, steps, input_size, hidden_size):
        super().__init__()
        self.steps = steps

        self.d_f = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, input_size)
        )

    def step_do(self, state):
        x = state

        _ = self.d_f(x)

        step_out = x + torch.clamp(_, -1e5, 1e5)
        return step_out, step_out

    def forward(self, init, steps=None):
        state = init
        outputs = []

        steps = self.steps if steps is None else steps

        for step in range(steps):
            step_out, state = self.step_do(state)
            outputs.append(step_out)

        outputs = torch.stack(outputs, dim=1)

        return outputs


mse_loss = torch.nn.MSELoss(reduction='none')


def ode_loss(y_true, y_pred):
    mask = torch.sum(y_true, dim=-1, keepdim=True) > 0
    mask = mask.float()

    return torch.sum(mask * mse_loss(y_true, y_pred)) / mask.sum()


if __name__ == '__main__':

    steps, h = 50, 1

    series = {0: [100, 150],
              10: [165, 283],
              15: [197, 290],
              30: [280, 276],
              36: [305, 269],
              40: [318, 266],
              42: [324, 264]}

    X = np.array([series[0]])
    Y = np.zeros((1, steps, 2))

    for i, j in series.items():
        if i != 0:
            Y[0, int(i / h) - 1] += series[i]

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    model = ODE_RNN(steps, input_size=2, hidden_size=64)
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(10000):
        outputs = model(X)
        loss = ode_loss(Y, outputs)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(epoch, loss.item())

    with torch.no_grad():
        result = model(torch.tensor([[100, 150]], dtype=torch.float32))[0]
        times = np.arange(1, steps + 1) * h

    plot.clf()
    plot.plot(times, result[:, 0], color='blue')
    plot.plot(times, result[:, 1], color='green')

    plot.plot(list(series.keys()), [i[0] for i in series.values()], 'o', color='blue')
    plot.plot(list(series.keys()), [i[1] for i in series.values()], 'o', color='green')
    plot.savefig('ode_b.png')
