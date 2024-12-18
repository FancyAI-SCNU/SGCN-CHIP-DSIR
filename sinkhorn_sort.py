import torch
import math


def cost_fn(x, y, power):
    """
    Function to compute transportation cost as a |x - y|^power

    Parameters
    ----------
    a : np.array, dims = (N_x)
         Marginal distribution of x's

    x : np.array, dims = (N_x)
        Numbers to be sorted,  (default is 0.1)

    power : int, optional
        Function representing distance between two elements. Its expected to be convex (default is L2)
    """
    delta = x[:, None] - y[None, :]
    if power == 1.0:
        cost = torch.abs(delta)
    elif power == 2.0:
        cost = delta ** 2.0
    else:
        abs_diff = torch.abs(delta)
        cost = abs_diff ** power
    return cost



def get_distr(u, v, K):
    """
    Given a, b functions returns the distribution of mass
    
    Parameters
    ----------
    a : np.array, dims = (N_x)
        Vector a computed by Sinkhorn
        
    b : np.array, dims = (N_y)
        Vector b computed by Sinkhorn

    K : np.array, dims = (N_x, N_y)
        The kernel function of pairs of objects
        
    """

    return (torch.diag(u).dot(K)).dot(torch.diag(v))


def Sinkhorn(a, b, x, y, eps, power, nu):
    """
    The Sinkhorn algorithm to compute u, v functions
    
    This is the numericaly unstable version, does not work with eps <= 0.01.
    
    Parameters
    ----------
    a : np.array, dims = (N_x)
         Marginal distribution of x's
        
    b : np.array, dims = (N_y)
        Marginal distribution of y's
        
    X : np.array, dims = (N_x)
        Numbers to be sorted,  (default is 0.1)
        
    Y : np.array, dims = (N_y)
        Milestones to be compared against, its expected to be equaly space from 0 to 1
        
    eps : float,
        Strength of regularisation 
        
    power : float,
        Function representing distance between two elements. Its expected to be convex
        
    nu : float,
        Error to tolerate for convergance
    """

    # size_x = x.shape[0]
    # size_y = y.shape[0]
    C = cost_fn(x, y, power)

    K = torch.exp(-C / eps)
    u = torch.ones_like(x)

    v = b / (K.T @ u)
    u = a / (K @ v)

    while torch.abs(v * (K.T @ u) - b).sum() > nu:
        v = b / (K.T @ u)
        u = a / (K @ v)

    return u, v, K


def Rank_Sort(a, b, x, y, eps=0.1, power=2, nu=1e-5):
    """
    Function to get Ranking and Sorted values
    
    This is the numericaly unstable version, does not work with eps <= 0.01.
    
    Parameters
    ----------
    a : np.array, dims = (N_x)
         Marginal distribution of x's
        
    x : np.array, dims = (N_x)
        Numbers to be sorted,  (default is 0.1)
        
    b : np.array, dims = (N_y)
        Marginal distribution of y's
        
    y : np.array, dims = (N_y)
        Milestones to be compared against, its expected to be equaly space from 0 to 1
        
    eps : float, optional
        Strength of regularisation (default is 0.1)
        
    power : int, optional
        Function representing distance between two elements. Its expected to be convex (default is L2)
        
    nu : float, optional
        Error to tolerate for convergance (default is 1e-5)
    """

    u, v, K = Sinkhorn(a, b, x, y, eps, power, nu)
    b_hat = torch.cumsum(b, dim=0)  # ??
    n = x.size(0)

    R_tilda = (n * (a ** -1)) * u * (K @ (v * b_hat))
    S_tilda = (b ** -1) * v * (K.T @ (u * x))

    return R_tilda, S_tilda


def get_delta(cost, alpha, betta, b, eps):
    """
    This function returns the distance between current approximation of Sinkhorn and true distribution
    
    Parameters
    ----------
    cost : np.array, dims = (N_x, N_y)
        Transportation cost, computed by cost_fn
        
    alpha : np.array, dims = (N_x)
        alpha computed by log_Sinkhorn
        
    betta : np.array, dims = (N_x)
        betta computed by log_Sinkhorn
    
    b : np.array, dims = (N_y)
        Marginal distribution of y's
        
    eps : float,
        Strength of regularisation
    """
    b_bar = torch.exp(-(cost.T - alpha.T - betta) / eps).sum(dim=1)
    return torch.abs(b - b_bar).sum()


def log_Sinkhorn(a, b, x, y, eps=1e-2, power=2, nu=1e-5):
    """
    The Sinkhorn algorithm to compute u, v functions
    
    This is the numericaly stable version, does not work with eps <= 0.00001.
    
    Parameters
    ----------
    a : np.array, dims = (N_x)
         Marginal distribution of x's
        
    b : np.array, dims = (N_y)
        Marginal distribution of y's
        
    X : np.array, dims = (N_x)
        Numbers to be sorted,  (default is 0.1)
        
    Y : np.array, dims = (N_y)
        Milestones to be compared against, its expected to be equaly space from 0 to 1
        
    eps : float,
        Strength of regularisation 
        
    power : float,
        Function representing distance between two elements. Its expected to be convex
        
    nu : float,
        Error to tolerate for convergance
    """
    C = cost_fn(x, y, power)

    alpha = torch.zeros((x.shape[0], 1)).to(x.device)
    betta = torch.zeros((y.shape[0], 1)).to(x.device)

    while get_delta(C, alpha, betta, b, eps) > nu:
        alpha = eps * torch.log(a) + soft_min(C - alpha - betta.T, eps) + alpha
        betta = eps * torch.log(b) + soft_min(C.T - alpha.T - betta, eps) + betta

    return alpha, betta, C


def Id(x):
    """
    Identity functinon
    
    Parameters
    ----------
    x : np.array, dims = (N_x)
    """
    return x


# def squash(x, scale=1.0, min_std=1e-10):
def squash(x, p1, min_std):
    """
    Function to map x into [0, 1]
    
    Parameters
    ----------
    x : np.array, dims = (N_x)
    """
    mu = torch.mean(x)
    s = p1 * torch.maximum(torch.std(x), min_std)
    return 1 / (1 + torch.exp(-((x - mu) / s)))


def soft_min(M, eps):
    """
    Linewise soft min operator
    
    Parameters
    ----------
    x : np.array, dims = (N_x, N_y)
    """
    return -eps * torch.log(torch.exp(-M / eps).sum(dim=1, keepdim=True))


def Rank_Sort_log(a, b, x, y, eps=1e-2, power=2, nu=1e-5, param1=None, param2=None):
    """
    Function to get Ranking and Sorted values
    
    This is the numerically unstable version, does not work with eps <= 0.0001.
    
    Parameters
    ----------
    a : np.array, dims = (N_x)
         Marginal distribution of x's
        
    x : np.array, dims = (N_x)
        Numbers to be sorted,  (default is 0.1)
        
    b : np.array, dims = (N_y)
        Marginal distribution of y's
        
    y : np.array, dims = (N_y)
        Milestones to be compared against, its expected to be equaly space from 0 to 1
        
    eps : float, optional
        Strength of regularisation (default is 0.1)
        
    power : int, optional
        Function representing distance between two elements. Its expected to be convex (default is L2)
        
    nu : float, optional
        Error to tolerate for convergance (default is 1e-5)
    """

    gx = squash(x, param1, param2)
    alpha, betta, C = log_Sinkhorn(a, b, gx, y, eps, power, nu)
    b_hat = torch.cumsum(b, dim=0)

    R_tilda = len(x) * (a ** -1) * (torch.exp(-(C - alpha - betta.T) / eps) @ b_hat)
    S_tilda = (b ** -1) * torch.exp(-(C.T - alpha.T - betta) / eps) @ x[:, None]

    return R_tilda.flatten(), S_tilda.flatten()
