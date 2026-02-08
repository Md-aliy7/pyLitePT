import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from models.utils import offset2batch


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        result = {}
        for key in batch[0]:
            if "gt_boxes" in key:
                 # Special handling for gt_boxes: Pad and Stack to (B, N_max, C)
                 boxes_list = [d[key] for d in batch]
                 max_boxes = max([b.shape[0] for b in boxes_list])
                 box_dim = boxes_list[0].shape[1]
                 batch_boxes = torch.zeros((len(boxes_list), max_boxes, box_dim), dtype=boxes_list[0].dtype)
                 for i, boxes in enumerate(boxes_list):
                     batch_boxes[i, :boxes.shape[0], :] = boxes
                 result[key] = batch_boxes
            elif "offset" in key:
                # offset -> bincount -> concat bincount-> concat offset
                result[key] = torch.cumsum(
                    collate_fn([d[key].diff(prepend=torch.tensor([0])) for d in batch]),
                    dim=0,
                )
            else:
                result[key] = collate_fn([d[key] for d in batch])
        
        # Fix: Generate offset if not present (required by Point structure)
        if "coord" in result and "offset" not in result:
            lengths = [d["coord"].shape[0] for d in batch]
            result["offset"] = torch.cumsum(torch.tensor(lengths, dtype=torch.long), dim=0)
        
        # Generate 'batch' from 'offset' (required by Point structure)
        if "offset" in result and "batch" not in result:
            result["batch"] = offset2batch(result["offset"])
        
        # Generate 'feat' from coord+color (required by Point structure for sparsify)
        if "coord" in result and "feat" not in result:
            if "color" in result:
                # Feat = [coord, color] = 6 channels
                result["feat"] = torch.cat([result["coord"], result["color"]], dim=1)
            else:
                # Feat = coord = 3 channels
                result["feat"] = result["coord"].clone()
        
        # Add default grid_size if not present (required for sparsify)
        if "coord" in result and "grid_size" not in result:
            result["grid_size"] = torch.tensor([0.02])  # Default 2cm voxel size
        
        # Compute grid_coord if not present
        if "coord" in result and "grid_coord" not in result and "grid_size" in result:
            coord = result["coord"]
            grid_size = result["grid_size"]
            if isinstance(grid_size, torch.Tensor):
                grid_size = grid_size.item() if grid_size.numel() == 1 else grid_size[0].item()
            
            # Compute coord offset for grid_coord only (not for coord itself)
            coord_min = coord.min(0)[0]
            
            result["grid_coord"] = torch.div(
                coord - coord_min, grid_size, rounding_mode="trunc"
            ).int()
            
            # NOTE: coord and gt_boxes remain in original coordinates
            # Only grid_coord is shifted for sparse convolution indexing
            
        return result
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if random.random() < mix_prob:
        if "instance" in batch.keys():
            offset = batch["offset"]
            start = 0
            num_instance = 0
            for i in range(len(offset)):
                if i % 2 == 0:
                    num_instance = max(batch["instance"][start : offset[i]])
                if i % 2 != 0:
                    mask = batch["instance"][start : offset[i]] != -1
                    batch["instance"][start : offset[i]] += num_instance * mask
                start = offset[i]
        if "offset" in batch.keys():
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )

        ### fix bug ###
        # recompute grid coord !!
        grid_coord_new = []
        batch_size = len(batch["offset"])

        batch_mask = offset2batch(batch["offset"])
        for bs_id in range(batch_size):
            sample_mask = batch_mask == bs_id
            coord_sample = batch['coord'][sample_mask]
            scaled_coord_sample = coord_sample / batch['grid_size'][0]  # hack here! 
            grid_coord_sample = torch.floor(scaled_coord_sample).to(torch.int64)
            min_coord_sample= grid_coord_sample.min(0)[0]
            grid_coord_sample -= min_coord_sample

            grid_coord_new.append(grid_coord_sample)

        grid_coord_new = torch.cat(grid_coord_new, dim=0)
        batch["grid_coord"] = grid_coord_new
        ### fix bug ###

    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))
