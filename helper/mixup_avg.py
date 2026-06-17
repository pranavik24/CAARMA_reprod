import torch 

def mixup_data_euc_avg(x, W, labels):
    batch_size = x.size()[0]
    index = []
    w_mix = torch.zeros(W.size(0), batch_size, device=W.device, dtype=W.dtype)
    y_mix = torch.zeros(batch_size, dtype=torch.int64)
    set_label = [int(label) for label in torch.unique(labels.detach()).cpu().tolist()]

    if len(set_label) <= 1:
        speaker = set_label[0]
        return x, y_mix.to(x.device), W[:, speaker:speaker + 1].to(x.device)

    dic_spk = {}
    for single_spk in set_label:
        candidate_speakers = [speaker for speaker in set_label if speaker != single_spk]
        distances = torch.stack([
            torch.dist(W[:, single_spk], W[:, speaker])
            for speaker in candidate_speakers
        ])
        closest_neighbor_index = int(torch.argmin(distances).item())
        dic_spk[single_spk] = candidate_speakers[closest_neighbor_index]

    lst_labels = labels.tolist()
    newlabel = {}
    labelid = 0
    for i in range(batch_size):
        l1 = labels[i].item()
        l2 = dic_spk[l1] 
        dictidx = (int(l1), int(l2))
        if dictidx not in newlabel:
            newlabel[dictidx] = labelid
            w_mix[:,labelid] = (W[:, l1] + W[:, l2])/2
            labelid = labelid + 1
        else:
            w_mix[:,newlabel[dictidx]] = (W[:, l1] + W[:, l2])/2
        y_mix[i] = newlabel[dictidx]
        index.append(lst_labels.index(l2))
    x_mix = 0.5*(x + x[index,:])
    
    x_combined = x_mix
    w_combined = w_mix[:, 0:labelid].to(x.device)
    y_combined = y_mix.to(x.device)
    return x_combined, y_combined , w_combined
