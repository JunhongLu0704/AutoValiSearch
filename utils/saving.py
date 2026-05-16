import glob
import os

import torch


def save_checkpoint(state, is_best, log_path, epoch=0):
    if not is_best:
        return
    prefix = 'model_best_at'
    files_to_delete = glob.glob(os.path.join(os.path.dirname(log_path), f"{prefix}*"))
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"remove : {file_path}")
        except OSError as e:
            print(f"remove: {file_path}, failed: {e}")
    best_path = os.path.join(os.path.dirname(log_path), f'model_best_at_{epoch}.pth')
    torch.save(state, best_path)
    print(f"save: best model at {epoch}")
    # shutil.copyfile(savename, best_path)
