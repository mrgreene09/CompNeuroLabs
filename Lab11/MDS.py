from __future__ import division

def cmdscale(D):

    import numpy as np

    # Number of points                                                                        
    n = len(D)
 
    # Centering matrix                                                                        
    matrix = np.eye(n) - np.ones((n, n))/n
 
    # YY^T                                                                                    
    B = -matrix.dot(D**2).dot(matrix)/2
 
    # Diagonalize                                                                             
    evals, evecs = np.linalg.eigh(B)
 
    # Sort by eigenvalue in descending order                                                  
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
 
    # Compute the coordinates                     
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)
 
    return Y, evals