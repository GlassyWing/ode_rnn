import torch.nn as nn
import torch
from torch.optim import Adam, SGD
import numpy as np
import matplotlib.pyplot as plot


class ODE_RNN(nn.Module):

    def __init__(self, steps, h):
        super().__init__()
        self.steps = steps
        self.h = h

        self.weights = nn.Parameter(
            torch.tensor([0.1, 0.1, 0.001, 0.001, 0.001, 0.001], dtype=torch.float32)
            , requires_grad=True)

    def step_do(self, state):
        x = state
        r1, r2, a1, a2, iN1, iN2 = (self.weights[0], self.weights[1],
                                    self.weights[2], self.weights[3],
                                    self.weights[4], self.weights[5])
        _1 = r1 * x[:, 0] * (1 - iN1 * x[:, 0]) - a1 * x[:, 0] * x[:, 1]
        _2 = r2 * x[:, 1] * (1 - iN2 * x[:, 1]) - a2 * x[:, 0] * x[:, 1]

        _ = torch.stack((_1, _2), dim=-1)

        step_out = x + self.h * torch.clamp(_, -1e5, 1e5)
        return step_out, step_out

    def forward(self, init):
        state = init
        outputs = []
        for step in range(self.steps):
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

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    model = ODE_RNN(steps, h)
    optimizer = Adam(model.parameters(), lr=1e-4)

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
    plot.savefig('ode_a.png')
