import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA, SparsePCA
from scipy.linalg import svd

def statistical_factor_model(R, tau_s, tau_f, p, m):
    """
    Estimate a statistical factor model with time-series reweighting and two-stage SVD.
    
    Args:
        R (np.ndarray): Asset returns matrix (n x T).
        tau_s (float): Idiosyncratic reweighting decay parameter (tau_s).
        tau_f (float): First-stage decay parameter (tau_f).
        p (int): Number of factors in first-stage SVD.
        m (int): Number of factors in final model.
    
    Returns:
        B (np.ndarray): Factor loadings (n x m).
        F (np.ndarray): Factor scores (m x T).
        sigma (np.ndarray): Idiosyncratic volatilities (n x 1).
    """
    # 1. Input validation
    assert tau_s >= tau_f > 0
    n, T = R.shape
    
    # 2. First-stage reweighting
    weights_tau_f = np.exp(-np.arange(T, 0, -1) / tau_f)
    weights_tau_f *= T / np.sum(weights_tau_f)
    W_tau_f = np.diag(weights_tau_f)
    R_tilde = R @ W_tau_f

    # 3. First-stage PCA
    U_tilde, S_tilde, Vt_tilde = np.linalg.svd(R_tilde, full_matrices=False)
    U_p = U_tilde[:, :p]
    S_p = np.diag(S_tilde[:p])
    V_p = Vt_tilde[:p, :]
    
    # 4. Idiosyncratic proxy estimation
    E = R_tilde - U_p @ S_p @ V_p
    sigma_sq = np.mean(E**2, axis=1)
    sigma = np.sqrt(sigma_sq)
    W_sigma = np.diag(1 / sigma)
    
    # 5. Idiosyncratic reweighting
    weights_tau_s = np.exp(-np.arange(T, 0, -1) / tau_s)
    weights_tau_s *= T / np.sum(weights_tau_s)
    W_tau_s = np.diag(weights_tau_s)
    R_hat = W_sigma @ R @ W_tau_s
    
    # 6. Second-stage PCA
    U_hat, S_hat, Vt_hat = np.linalg.svd(R_hat, full_matrices=False)
    
    # 7. Extract top m factors
    U_m = U_hat[:, :m]
    S_m = np.diag(S_hat[:m])
    V_m = Vt_hat[:m, :]
    
    # 8. Compute outputs
    B = np.linalg.inv(W_sigma) @ U_m
    B /= np.linalg.norm(B, axis=0) 
    
    return B,W_sigma


def statistical_factor_model_ica(R, tau_s, tau_f, p, m, random_state=0):
    """
    Statistical factor model with ICA in second stage.
    
    Args:
        R (np.ndarray): Asset returns matrix (n x T).
        tau_s (float): Idiosyncratic reweighting decay parameter.
        tau_f (float): First-stage decay parameter.
        p (int): Number of factors in first-stage SVD.
        m (int): Number of factors in final ICA model.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        B (np.ndarray): Factor loadings (n x m).
        F (np.ndarray): Factor scores (m x T).
        sigma (np.ndarray): Idiosyncratic volatilities (n x 1).
    """
    assert tau_s >= tau_f > 0
    n, T = R.shape
    
    # --- First-stage time weighting ---
    weights_tau_f = np.exp(-np.arange(T, 0, -1) / tau_f)
    weights_tau_f *= T / np.sum(weights_tau_f)
    W_tau_f = np.diag(weights_tau_f)
    R_tilde = R @ W_tau_f

    # --- First-stage PCA ---
    U_tilde, S_tilde, Vt_tilde = np.linalg.svd(R_tilde, full_matrices=False)
    U_p = U_tilde[:, :p]
    S_p = np.diag(S_tilde[:p])
    V_p = Vt_tilde[:p, :]

    # --- Idiosyncratic volatility estimation ---
    E = R_tilde - U_p @ S_p @ V_p
    sigma_sq = np.mean(E**2, axis=1)
    sigma = np.sqrt(sigma_sq)
    W_sigma = np.diag(1 / sigma)

    # --- Second-stage time weighting ---
    weights_tau_s = np.exp(-np.arange(T, 0, -1) / tau_s)
    weights_tau_s *= T / np.sum(weights_tau_s)
    W_tau_s = np.diag(weights_tau_s)
    R_hat = W_sigma @ R @ W_tau_s

    # --- ICA on reweighted returns ---
    ica = FastICA(n_components=m, random_state=random_state)
    F = ica.fit_transform(R_hat.T).T   # (m x T)
    A = ica.mixing_                    # (n x m)
    
    # --- Scale loadings back ---
    B = np.linalg.inv(W_sigma) @ A
    B /= np.linalg.norm(B, axis=0) 

    return B, W_sigma

def statistical_factor_model_spca(R, tau_s, tau_f, p, m, alpha=1.0, random_state=0):
    """
    Statistical factor model with Sparse PCA in second stage.
    
    Args:
        R (np.ndarray): Asset returns matrix (n x T).
        tau_s (float): Idiosyncratic reweighting decay parameter.
        tau_f (float): First-stage decay parameter.
        p (int): Number of factors in first-stage PCA.
        m (int): Number of factors in final Sparse PCA model.
        alpha (float): Sparsity controlling parameter in Sparse PCA.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        B (np.ndarray): Factor loadings (n x m).
        F (np.ndarray): Factor scores (m x T).
        sigma (np.ndarray): Idiosyncratic volatilities (n x 1).
    """
    assert tau_s >= tau_f > 0
    n, T = R.shape

    # --- First-stage reweighting ---
    weights_tau_f = np.exp(-np.arange(T, 0, -1) / tau_f)
    weights_tau_f *= T / np.sum(weights_tau_f)
    W_tau_f = np.diag(weights_tau_f)
    R_tilde = R @ W_tau_f

    # --- First-stage PCA ---
    U_tilde, S_tilde, Vt_tilde = np.linalg.svd(R_tilde, full_matrices=False)
    U_p = U_tilde[:, :p]
    S_p = np.diag(S_tilde[:p])
    V_p = Vt_tilde[:p, :]

    # --- Idiosyncratic volatility estimation ---
    E = R_tilde - U_p @ S_p @ V_p
    sigma_sq = np.mean(E**2, axis=1)
    sigma = np.sqrt(sigma_sq)
    W_sigma = np.diag(1 / sigma)

    # --- Second-stage reweighting ---
    weights_tau_s = np.exp(-np.arange(T, 0, -1) / tau_s)
    weights_tau_s *= T / np.sum(weights_tau_s)
    W_tau_s = np.diag(weights_tau_s)
    R_hat = W_sigma @ R @ W_tau_s  # n x T

    # --- Sparse PCA ---
    sparse_pca = SparsePCA(n_components=m, alpha=alpha, random_state=random_state)
    F = sparse_pca.fit_transform(R_hat.T).T   # m x T
    B_sparse = sparse_pca.components_.T       # n x m

    # --- Rescale loadings back ---
    B = np.linalg.inv(W_sigma) @ B_sparse
    B /= np.linalg.norm(B, axis=0) 

    return B, W_sigma