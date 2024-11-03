
FIXEXPM = 15
FIXEXPL = 10
FIXSHRIK = -15

def fixSp(imgTensor):
    W = imgTensor.shape[-1]
    pos_block_w = [pos *  (W // 4) for pos in range(5)]
    img_1 = imgTensor[:, :, pos_block_w[0] + FIXEXPM: pos_block_w[1] + FIXEXPM]
    img_2 = imgTensor[:, :, pos_block_w[1] + FIXEXPM: pos_block_w[2] + FIXEXPL]
    img_3 = imgTensor[:, :, pos_block_w[2] + FIXEXPL: pos_block_w[3]]
    img_4 = imgTensor[:, :, pos_block_w[3]: FIXSHRIK]
    return img_1, img_2, img_3, img_4