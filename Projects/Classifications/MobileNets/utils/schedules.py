#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
from __future__ import print_function
import math
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
import numpy as np
import copy

def plot_schedule(schedule_fn, iterations=1500):
    # Iteration count starting at 1
    iterations = [i+1 for i in range(iterations)]
    lrs = [schedule_fn(i) for i in iterations]
    plt.style.use('ggplot')
    plt.scatter(iterations, lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.show()

class TrianguarSchedule:
    def __init__(self, min_lr, max_lr, cycle_length, in_fraction=0.5):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.end_increase = in_fraction * cycle_length
        
    def __call__(self, iteration):
        if iteration < self.end_increase:
            unit_cycle = iteration / self.end_increase
        elif iteration < self.cycle_length:
            unit_cycle = 1 - (iteration - self.end_increase) / (self.cycle_length - self.end_increase)
        else:
            unit_cycle = 0
        curr_lr = unit_cycle * (self.max_lr - self.min_lr) + self.min_lr
        return curr_lr

import math
class CosineAnnealingSchedule:
    def __init__(self, min_lr, max_lr, cycle_length):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length

    def __call__(self, iteration):
        if iteration < self.cycle_length:
            return (self.max_lr + self.min_lr)/2 + (self.max_lr - self.min_lr)/2 * math.cos(iteration/self.cycle_length * math.pi)
        else:
            return self.min_lr

class CyclicalSchedule:
    def __init__(self, schedule_class, cycle_length, cycle_length_decay=1, cycle_magnitude_decay=1, **kwags):
        """
        schedule_class: class of schedule, expected to take `cycle_length` argument.
        cycle_length: iterations used for initial cycle (int)
        cycle_length_decay: factor multiplied to cycle_length each cycle (float)
        cycle_magnitude_decay: factor multiplied learning rate magnitudes each cycle (float)
        kwargs: passed to the schedule_class
        
        """
        self.schedule_class = schedule_class
        self.length = cycle_length
        self.length_decay = cycle_length_decay
        self.magnitude_decay = cycle_magnitude_decay
        self.kwags = kwags
        
    def __call__(self, iteration):
        end_curr_cycle = self.length
        cycle_length = self.length  # 保存当前的cycle_length
        cycle_idx = 0
        # 每当iteration超出当前cycle end(即第二个cycle起)，将cycle_end延后一个cycle_length
        while iteration >= end_curr_cycle:
            cycle_length = cycle_length * self.length_decay
            cycle_idx += 1  # 用于decay cycle_magnitude
            end_curr_cycle += cycle_length
        cycle_offset = iteration - (end_curr_cycle - cycle_length)
        
        schedule = self.schedule_class(cycle_length=cycle_length, **self.kwags)
        return schedule(cycle_offset) * self.magnitude_decay**cycle_idx
