import random
import numpy as np

class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        total = len(self.memory)
        indices = np.random.choice(total, batch_size)
        
        return [self.memory[idx] for idx in indices], indices, [1/batch_size for _ in indices]

    def __len__(self):
        return len(self.memory)
        