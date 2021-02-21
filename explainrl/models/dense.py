import torch
from ..environment.state import GridState


class FeedForwardEvaluator(torch.nn.Module):

    def __init__(self, grid_size=(6, 6)):
        super(FeedForwardEvaluator, self).__init__()
        num_grid_cells = grid_size[0] * grid_size[1]
        self.linear_1 = torch.nn.Linear(num_grid_cells * 3, 100)
        self.linear_2 = torch.nn.Linear(100, 50)
        self.value_linear = torch.nn.Linear(50, 1)
        self.policy_linear = torch.nn.Linear(50, 4)

    def forward(self, state):
        x = self.representation(state)
        x = x.view(-1)
        x = torch.nn.ELU(self.linear_1(x))
        x = torch.nn.ELU(self.linear_2(x))
        value = self.value_linear(x)
        policy = torch.nn.Softmax(self.policy_linear(x))
        return value, policy

    @staticmethod
    def representation(state: GridState):
        layout = torch.from_numpy(state.grid).float()
        tiles = torch.zeros(layout.shape).float()
        targets = torch.zeros(layout.shape).float()
        for x, y in state.tiles:
            tiles[x, y] = 1
        for x, y in state.targets:
            targets[x, y] = 1
        representation = torch.stack([layout, tiles, targets])
        return representation
