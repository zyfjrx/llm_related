import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


class MoELayer(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Linear(in_features, out_features) for _ in range(num_experts)])
        self.gate = nn.Linear(in_features, num_experts)

    def forward(self, x):
        gate_score = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts],dim=1)
        output = torch.bmm(gate_score.unsqueeze(1), expert_outputs).squeeze(1)
        return output

if __name__ == '__main__':
    num_experts = 4
    in_features = 8
    out_features = 3
    batch_size = 2
    model = MoELayer(num_experts, in_features, out_features)
    x = torch.randn(batch_size,in_features) #[2,8]
    out = model(x)
    print(out.shape)
    print(out)

