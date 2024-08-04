import torch
import numpy as np
import math

    
def transl_error(predictions, targets):
    errors = []
    for pred, target in zip(predictions, targets):
        pred_transl = pred[:3]
        target_transl = target[:3]
        # Compute L2 error
        error = torch.norm(pred_transl - target_transl)
        errors.append(error)
    return errors

def rot_error(predictions, targets):
    errors = []
    for pred, target in zip(predictions, targets):
        # Compute rotational error in degrees using the same formula as the bop toolkit
        pred_rot = pred[3:].reshape(3, 3).cpu().numpy()
        target_rot = target[3:].reshape(3, 3).cpu().numpy()
        error_cos = float(0.5 * (np.trace(pred_rot.dot(np.linalg.inv(target_rot))) - 1.0))
        # Avoid invalid values due to numerical errors.
        error_cos = min(1.0, max(-1.0, error_cos))
        error = math.acos(error_cos)
        error = error * 180/np.pi 
        errors.append(error)
    return errors

def recall(errors, threshold):
    return len([error for error in errors if error < threshold]) / len(errors)