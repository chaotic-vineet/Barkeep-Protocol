import torch

from src.model_files.definitions import OptimizerConfig

class AdamW:
    def __init__(self, parameters, cfg: OptimizerConfig):
        self.parameters = list(parameters)
        self.first_moments = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.second_moments = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.beta1, self.beta2 = cfg.betas
        self.eps = cfg.eps
        self.weight_decay = cfg.weight_decay
        self.t = 1

    def __call__(self, lr):
        for idx, parameter in enumerate(self.parameters):
            first_moment = self.beta1 * self.first_moments[idx] + (1 - self.beta1) * parameter.grad
            self.first_moments[idx] = first_moment

            second_moment = self.beta2 * self.second_moments[idx] + (1 - self.beta2) * (parameter.grad**2)
            self.second_moments[idx] = second_moment

            first_moment_hat  = first_moment  / (1 - self.beta1 ** self.t)
            second_moment_hat = second_moment / (1 - self.beta2 ** self.t)

            parameter.data -= lr * first_moment_hat / (torch.sqrt(second_moment_hat) + self.eps)

            if parameter.ndim >= 2:
                parameter.data -= lr * self.weight_decay * parameter.data
        self.t+=1

    def step_magnitudes(self, lr) -> list:
        magnitudes = []
        for idx, parameter in enumerate(self.parameters):
            first_moment = self.first_moments[idx]
            second_moment = self.second_moments[idx]

            first_moment_hat  = first_moment  / (1 - self.beta1 ** (self.t-1))
            second_moment_hat = second_moment / (1 - self.beta2 ** (self.t-1))

            step = lr * first_moment_hat / (torch.sqrt(second_moment_hat) + self.eps)

            magnitudes.append(step)

        return magnitudes

    def save_state(self, path):
        torch.save({
            't': self.t,
            'first_moments': self.first_moments,
            'second_moments': self.second_moments,
        }, path)

    def load_state(self, path):
        state = torch.load(path)
        self.t = state['t']
        self.first_moments = state['first_moments']
        self.second_moments = state['second_moments']