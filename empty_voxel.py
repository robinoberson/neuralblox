import numpy as np
import os
import torch
import torch.optim as optim

from src import config, data, layers
from src.checkpoints import CheckpointIO
from src.common import define_align_matrix
from os.path import join
from tqdm import trange


cfg_path = '/home/roberson/MasterThesis/master_thesis/neuralblox/configs/pointcloud/backbone_0.6_1.6.yaml'
cfg_default_path = '/home/roberson/MasterThesis/master_thesis/neuralblox/configs/default.yaml'

cfg = config.load_config(cfg_path, cfg_default_path)
is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")

train_dataset = config.get_dataset('train', cfg)
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
vis_n_outputs = cfg['generation']['vis_n_outputs']
starting_model = cfg['training']['starting_model']


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=cfg['training']['n_workers'], shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

model = config.get_model(cfg, device=device, dataset=train_dataset)

generator = config.get_generator(model, cfg, device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model)
load_dict = checkpoint_io.load(starting_model)

it = 0
for batch in train_loader:
    print(batch.keys())
    it += 1
    loss = trainer.train_step(batch)
