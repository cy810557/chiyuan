def ohem_loss(batch_size, cls_pred, cls_target, loc_pred, loc_target):
    ohem_loss_cls = cross_entropy(cls_pred, cls_target)
    ohem_loss_cor = l2_loss(loc_pred, loc_target)
    loss = ohem_loss_cls + ohem_loss_cor 
    sorted_ohem_loss, idx = sorted(loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], batch_size)
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx = idx[:keep_num]
        ohem_loss_cls = ohem_loss_cls[keep_idx]
        ohem_loss_cor = ohem_loss_cor[keep_idx]
    cls_loss = ohem_loss_cls() / keep_num
    cor_loss = ohem_loss_cor() / keep_num
    return cls_loss, cor_loss
