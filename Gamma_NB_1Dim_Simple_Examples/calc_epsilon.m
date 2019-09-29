function Eps = calc_epsilon(p, alpha)
%     """
%     Calculate the epsilon accepted by Numpy's internal Marsaglia & Tsang
%     rejection sampler. (h is invertible)
%     """
    sqrtAlpha = sqrt(9. *alpha-3.);
    t = alpha-1./3;
%     tmp = sign(t).*max(abs(t),1e-4);
%     powZA = ( p./tmp ).^(1/3);
    powZA = ( p./t ).^(1/3);
    
    Eps = sqrtAlpha.*(powZA-1);