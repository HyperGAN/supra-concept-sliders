"""Validation plateau controller; a step limit never counts as convergence."""
from dataclasses import dataclass
import math


@dataclass
class Plateau:
    lr: float = 5e-5
    min_lr: float = 3.125e-6
    patience: int = 4
    final_patience: int = 6
    min_step: int = 2000
    relative_delta: float = 0.005
    stage_best: float = math.inf
    best: float = math.inf
    best_step: int = 0
    stale: int = 0
    reductions: int = 0

    def observe(self, step, score):
        if not math.isfinite(score) or score < 0:
            raise ValueError(f"Invalid convergence score: {score}")
        selected = score < self.best
        if selected:
            self.best, self.best_step = score, step
        if score < self.stage_best * (1 - self.relative_delta):
            self.stage_best, self.stale = score, 0
        else:
            self.stale += 1
        at_floor = self.lr <= self.min_lr * (1 + 1e-9)
        wait = self.final_patience if at_floor else self.patience
        action = "continue"
        if step >= self.min_step and self.stale >= wait:
            if at_floor:
                action = "converged"
            else:
                self.lr = max(self.min_lr, self.lr / 2)
                self.reductions += 1
                self.stale, self.stage_best = 0, score
                action = "reduce_lr"
        return action, selected
