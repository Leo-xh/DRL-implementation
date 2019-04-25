import random
import numpy as np


class PrioritizedExperienceReplayMemory:
    def __init__(self, capacity, alpha, beta_start=0.4, beta_frames=10000):
        self.prob_alpha = alpha
        self.capacity = capacity
        self.memory = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1
        self.beta_start = beta_start
        self.beta_frames = beta_frames

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, transition):
        max_prio = self.priorities.max() if self.memory else 1.0**self.prob_alpha

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        total = len(self.memory)
        probs = prios / prios.sum()
        indices = np.random.choice(total, batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        #min of ALL probs, not just sampled probs
        prob_min = probs.min()
        max_weight = (prob_min*total)**(-beta)
        weights = (total * probs[indices]) ** (-beta)
        weights /= max_weight

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = (prio + 1e-5)**self.prob_alpha

    def __len__(self):
        return len(self.memory)
