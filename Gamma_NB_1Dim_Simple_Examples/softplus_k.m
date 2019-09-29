function y = softplus_k(x,k)
%% softplus_k
% y = log_k (1 + k^x)
%   = log(1 + k^x) / log(k)
if ~exist('k','var')
    k = exp(1) ;
end

if k <= 0 
    error('k must not <= 0')
elseif k == 1
    error('k must not == 1')
else
    y = log1p(k.^x) / log(k) ; 
end