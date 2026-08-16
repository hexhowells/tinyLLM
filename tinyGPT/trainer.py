from collections import defaultdict

import time
from typing import Callable

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils import clip_grad_norm_


class Trainer:
    def __init__(self, config: dict, model, train_dataset):
        self.config = config
        self.model = model
        self.optimiser = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        if config['device'] == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config['device']
        
        self.model = self.model.to(self.device)
        print(f'Running on device: {self.device}')

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
    

    def add_callback(self, oneevent: str, callback: Callable) -> None:
        """
        Add callback to trainer for specific event

        Args:
            oneevent: event name to add callback to
            callback: callback function to add
        """
        self.callbacks[oneevent].append(callback)
    

    def set_callback(self, oneevent: str, callback: Callable) -> None:
        """
        Set callback event to specified callback function

        Removes any other callbacks linked to that event name

        Args:
            oneevent: event name to add callback to
            callback: callback function to add
        """
        self.callbacks[oneevent] = [callback]
    

    def trigger_callbacks(self, oneevent: str) -> None:
        """
        Run all callback functions for specific event

        Args:
            oneevent: event to trigger callbacks on
        """
        for callback in self.callbacks.get(oneevent, []):
            callback(self)

    
    def run(self):
        """Run the training loop for the configured trainer."""
        model, config = self.model, self.config

        self.optimiser = model.configure_optimizers(config)

        train_loader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            _, self.loss = model(x, y)

            model.zero_grad(set_to_none=True)
            self.loss.backward()
            clip_grad_norm_(model.parameters(), config['grad_norm_clip'])
            self.optimiser.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            t_now = time.time()
            self.iter_dt = t_now = self.iter_time
            self.iter_time = t_now

            if config['max_iters'] is not None and self.iter_num >= config['max_iters']:
                break
