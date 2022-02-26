import numpy as np

def get_lm_68_to_5(preds):
    lm = np.array(preds[0])
    lm_nose          = lm[30]
    lm_eye_left      = lm[36 : 42, :2]
    lm_eye_right     = lm[42 : 48, :2]
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    mouth_left   = lm[48]
    mouth_right  = lm[54]

    return np.array([eye_left, eye_right, lm_nose, mouth_left, mouth_right]).astype(np.int32)