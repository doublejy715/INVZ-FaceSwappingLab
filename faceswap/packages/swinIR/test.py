import torch

def do_SR(model, img_lq, scale):
    _, _, h_old, w_old = img_lq.size()
    h_pad = (h_old // 8 + 1) * 8 - h_old
    w_pad = (w_old // 8 + 1) * 8 - w_old
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
    output = output = model(img_lq)
    output = output[..., :h_old * scale, :w_old * scale]
    return output