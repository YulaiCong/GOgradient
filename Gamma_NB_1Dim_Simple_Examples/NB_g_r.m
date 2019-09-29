function y = NB_g_r(zsample,QR,QP)
% Calculate the function g_{r}(z) with respect to NB distribution
% shape parameter -- r
% 
% Yulai Cong
% 2018/01/24

% y = -(log(1-QP)-psi(QR)+psi(zsample+1+QR)) / ((1-QP)^QR * QP^zsample) ...
%     * (zsample+QR) * beta(QR,zsample+1) * betainc(1-QP,QR,zsample+1) ;
% y = y + (zsample+QR)/(QP^zsample * QR^2) * ...
%     hypergeom([QR,QR,-zsample],[QR+1,QR+1],1-QP) ;

y = (zsample+QR) ./ exp(zsample * log(QP)) * (...
    -((log(1-QP)-psi(QR)+psi(zsample+1+QR)) / ((1-QP)^QR) * ...
    beta(QR,zsample+1) * betainc(1-QP,QR,zsample+1)) ...
    + (hypergeomq([QR,QR,-zsample],[QR+1,QR+1],1-QP) ./ (QR^2)) ) ;


% syms zsample QR QP
% y = (zsample+QR) ./ exp(zsample * log(QP)) * (...
%     -((log(1-QP)-psi(QR)+psi(zsample+1+QR)) / ((1-QP)^QR) * ...
%     beta(QR,zsample+1) * betainc(1-QP,QR,zsample+1)) ...
%     + (hypergeom([QR,QR,-zsample],[QR+1,QR+1],1-QP) ./ (QR^2)) ) ;
% simplify(y)


