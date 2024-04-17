import torch

# implementation https://github.com/L1bra1/Self-Point-Flow/blob/main/pseudo_labels_utils/OT_RW_utils.py
def Cost_Gaussian_function(data1, data2, theta_2=3, threshold_2=12):
    # data1: (B, 3, N), data2: (B, seq, 3, N)
    seq = data2.shape[1]
    # data1 = data1[:, None, :, :]
    distance2_matrix = torch.sum(data1 ** 2, 2, keepdims=True).transpose(3, 2) # B, seq, N, 1
    distance2_matrix = distance2_matrix + torch.sum(data2 ** 2, 2, keepdims=True)  # B, seq, N, N
    distance2_matrix = distance2_matrix - 2 * torch.matmul(data1.transpose(3, 2), data2)
    # threshold
    
    support = distance2_matrix < threshold_2[None, :, None, None]
    distance_cost_matrix = 1 - torch.exp(-distance2_matrix / theta_2)
    return distance_cost_matrix, support

def Cost_cosine_distance(feature1, feature2):
    # input [B, 3, N], [B, 3, N]
    norm1 = feature1 / torch.linalg.norm(feature1, dim=1, keepdim=True)
    norm2 = feature2 / torch.linalg.norm(feature2, axis=1, keepdim=True)

    norm_matrix = torch.matmul(norm1, norm2.transpose(1, 0))
    norm_matrix = torch.abs(norm_matrix)
    norm_cost_matrix = (1 - norm_matrix)
    return norm_cost_matrix
    

def OT(C, epsilon=0.03, OT_iter=4):
    B, seq, N1, N2 = C.shape

    # Entropic regularisation
    K = torch.exp( -C / epsilon)

    # Init. of Sinkhorn algorithm
    a = torch.ones((B, seq, N1, 1), device=C.device, dtype=C.dtype) / N1
    prob1 = torch.ones((B, seq, N1, 1), device=C.device, dtype=C.dtype) / N1
    prob2 = torch.ones((B, seq, N2, 1), device=C.device, dtype=C.dtype) / N2

    # Sinkhorn algorithm
    for _ in range(OT_iter):
        # Update b
        KTa = torch.matmul(K.transpose(3, 2), a)
        b = prob2 / (KTa + 1e-8)

        # Update a
        Kb = torch.matmul(K, b)
        a = prob1 / (Kb + 1e-8)

    # Transportation map
    T = torch.multiply(torch.multiply(a, K), b.transpose(3, 2))
    return T
