function Gr = gamma_grad_logr(epsilon, alpha)
    
    b = alpha - 1./3.;
    c = 1./ sqrt(9.*b);
    v = 1.+epsilon.*c;
    
    Gr = 0.5./b - 9*epsilon.*(c.^3)./v;