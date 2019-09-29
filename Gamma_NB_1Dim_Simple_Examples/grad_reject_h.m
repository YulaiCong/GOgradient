function grad = grad_reject_h(epsilon, alpha)
    
%     Gradient of reparameterization without shape augmentation.
    
    b = max(alpha - 1./3.,0);
    c = 1 ./sqrt(9 .*b);
    v = 1.+epsilon.*c;
    
    grad = v.^3 - 13.5*epsilon.*b.*(v.^2).*(c.^3);