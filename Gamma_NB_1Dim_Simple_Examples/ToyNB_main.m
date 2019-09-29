% NB Toy Experiments
% 
% By Yulai Cong
% 2018.01.24
clear,clc,close all

LogReal = 1 ;
IsSave = 0 ;

IsAdam = 0 ;
Beta1Adam = 0.9 ;
Beta2Adam = 0.999 ;
epsilon = 1e-8 ;

if IsAdam
    LR = 0.9 ;
else
    LR = 0.1 ;
end
VarSampleNum = 20 ;
KLSamNum = 100 ;

MaxIter = 200 ;

%% Make Toy data
Chosen = 4 ; % 4 or 6
%%%           1     2     3    4    5   6   7   8
RSet     =  [1e-5, 0.01, 0.1, 0.5,  1, 10, 50, 100] ; 
xxmaxSet =  [ 5 ,   5,    5,   8,  10, 30, 80, 150] ;
LRset   =  [ 0,    0.2,  0,  0.1,0.01, 0.1, 0,  0] ;
if ~IsAdam
    LR = LRset(Chosen) ;
end
PSet = 0.2 * ones(1,20) ; 
xxminSet =  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] ;

PostR = RSet(Chosen) 
PostP = PSet(Chosen) 
xxmax = xxmaxSet(Chosen) ;
xxmin = xxminSet(Chosen) ;

xx = xxmin:xxmax ;

PostPDF = nbinpdf(xx,PostR,1-PostP) ;

yymax = max(PostPDF) ;

% figure(1),subplot(221),
% stem(xx,PostPDF,'r') ;
% axis([xxmin,xxmax,0,yymax])

%% BBVI + ADAM 
if 1
    for ChosenGrad =  { 'REINFORCE', 'GO', 'REINFORCE2'}   %  {'GO', 'REINFORCE'}
        ChosenGrad = ChosenGrad{1} ;
        % Initialize
        if LogReal
            QRLog = 0 ;
            QR = exp(QRLog) ;
            QPLog = log(0.5) ;
            QP = exp(QPLog) ;            
        else
            QR = 1 ;
            QP = 0.5 ;
        end
        KLDist = [] ;

        m0R = 0 ;  v0R = 0 ;
        m0P = 0 ;  v0P = 0 ;
        % Iteration
        for iter = 1:MaxIter

%             if iter > 1
%                 delete(h) ;
%             end

            for iii = 1:VarSampleNum

                zsample = nbinrnd(QR, 1-QP) ;
                
                switch ChosenGrad
                    case 'GO'
                        Dzfz = (log(zsample+PostR) + log(PostP)) - (log(zsample+QR) + log(QP)) ;
                        GGradR = NB_g_r(zsample,QR,QP) * Dzfz ;
                        GGradP = NB_g_p(zsample,QR,QP) * Dzfz ;
                    case 'REINFORCE'
                        fz = gammaln(zsample+PostR)-gammaln(PostR)+PostR*log(1-PostP)+zsample*log(PostP) ...
                             - (gammaln(zsample+QR)-gammaln(QR)+QR*log(1-QP)+zsample*log(QP)) ;
                        PRLogqz = psi(zsample+QR)-psi(QR)+log(1-QP) ;
                        PPLogqz = -QR/(1-QP) + zsample/QP ;
                        GGradR = fz * PRLogqz ;
                        GGradP = fz * PPLogqz ;
                    case 'REINFORCE2'
                        fz = gammaln(zsample+PostR)-gammaln(PostR)+PostR*log(1-PostP)+zsample*log(PostP) ...
                             - (gammaln(zsample+QR)-gammaln(QR)+QR*log(1-QP)+zsample*log(QP)) ;
                        PRLogqz = psi(zsample+QR)-psi(QR)+log(1-QP) ;
                        PPLogqz = -QR/(1-QP) + zsample/QP ;
                        GGradR = fz * PRLogqz ;
                        GGradP = fz * PPLogqz ;
                        
                        zsample = nbinrnd(QR, 1-QP) ;
                        fz = gammaln(zsample+PostR)-gammaln(PostR)+PostR*log(1-PostP)+zsample*log(PostP) ...
                             - (gammaln(zsample+QR)-gammaln(QR)+QR*log(1-QP)+zsample*log(QP)) ;
                        PRLogqz = psi(zsample+QR)-psi(QR)+log(1-QP) ;
                        PPLogqz = -QR/(1-QP) + zsample/QP ;
                        GGradR = (GGradR + fz * PRLogqz) / 2 ;
                        GGradP = (GGradP + fz * PPLogqz) / 2 ;
                    otherwise
                        error('Undefined Gradient!')
                end
                
                if isnan(GGradR) | isinf(GGradR)
                    a = 1 ;
                end
                
                if LogReal
                    GradRLog = GGradR * QR ;     
                    GradPLog = GGradP * QP * (1-QP) ;    
                    GradCun(iii,1) = GradRLog ;
                    GradCun(iii,2) = GradPLog ;
                else
                    GradCun(iii,1) = GGradR ;
                    GradCun(iii,2) = GGradP ;
                end
            end

            switch ChosenGrad
                case 'GO'
                    GO.GradMean(iter,:) = mean(GradCun,1) ;
                    GO.GradVar(iter,:) = var(GradCun,[],1) ;
                case 'REINFORCE'
                    REINFORCE.GradMean(iter,:) = mean(GradCun,1) ;
                    REINFORCE.GradVar(iter,:) = var(GradCun,[],1) ;
                case 'REINFORCE2'
                    REINFORCE2.GradMean(iter,:) = mean(GradCun,1) ;
                    REINFORCE2.GradVar(iter,:) = var(GradCun,[],1) ;
            end


            if LogReal
                GradRLog = GGradR * QR ; 
                GradPLog = GGradP * QP * (1-QP) ;
                
                % ADAM or PureGradient
                if IsAdam
                    m0R = Beta1Adam*m0R + (1-Beta1Adam)*GradRLog ;
                    v0R = Beta2Adam*v0R + (1-Beta2Adam)*GradRLog^2 ;
                    m0P = Beta1Adam*m0P + (1-Beta1Adam)*GradPLog ;
                    v0P = Beta2Adam*v0P + (1-Beta2Adam)*GradPLog^2 ;
                    
                    m0RHat = m0R/(1-Beta1Adam^iter) ;
                    v0RHat = v0R/(1-Beta2Adam^iter) ;
                    m0PHat = m0P/(1-Beta1Adam^iter) ;
                    v0PHat = v0P/(1-Beta2Adam^iter) ;
                    
                    QRLog = QRLog + LR * m0RHat / (sqrt(v0RHat)+epsilon)  ;
                    QPLog = QPLog + LR * m0PHat / (sqrt(v0PHat)+epsilon)  ;
                else
                    QRLog = QRLog + LR * GradRLog ;
                    QPLog = QPLog + LR * GradPLog ;
                end
                QR = exp(QRLog) ;
                QP = 1/(1+exp(-QPLog)) ;

            else
                % ADAM or PureGradient
                if IsAdam
                    m0R = Beta1Adam*m0R + (1-Beta1Adam)*GGradR ;
                    v0R = Beta2Adam*v0R + (1-Beta2Adam)*GGradR^2 ;
                    m0P = Beta1Adam*m0P + (1-Beta1Adam)*GGradP ;
                    v0P = Beta2Adam*v0P + (1-Beta2Adam)*GGradP^2 ;
                    
                    m0RHat = m0R/(1-Beta1Adam^iter) ;
                    v0RHat = v0R/(1-Beta2Adam^iter) ;
                    m0PHat = m0P/(1-Beta1Adam^iter) ;
                    v0PHat = v0P/(1-Beta2Adam^iter) ;
                    
                    QR = softplus_k(QR + LR * m0RHat / (sqrt(v0RHat)+epsilon) ,  1e4) ;
                    QP = softplus_k(QP + LR * m0PHat / (sqrt(v0PHat)+epsilon) ,  1e4) ;
                else
                    QR = softplus_k(QR + LR * GGradR,  1e4) ;
                    QP = softplus_k(QP + LR * GGradP,  1e4) ;
                end
            end

            if isnan(QR) | isinf(QR)
                a = 1 ;
            end
            
            % Evaluate with iterations
            KLTmp = 0 ; 
            for ii = 1:KLSamNum
                zsample = nbinrnd(QR, 1-QP) ;
                KLTmp = KLTmp +  1/KLSamNum * -1 * (gammaln(zsample+PostR)-gammaln(PostR)+PostR*log(1-PostP)+zsample*log(PostP) ...
                             - (gammaln(zsample+QR)-gammaln(QR)+QR*log(1-QP)+zsample*log(QP))) ;
            end
            KLDist(iter) = KLTmp ;
            fprintf('Iter = %4d, KLDist = [%9.2d], QR = %.2d (%.2d), QP = %.2d (%.2d) \n', iter,KLDist(iter), QR, PostR, QP, PostP)

            switch ChosenGrad
                case 'GO'
                    GO.KLDist = KLDist ;
                    GO.QR(iter) = QR ;
                    GO.QP(iter) = QP ;
                case 'REINFORCE'
                    REINFORCE.KLDist = KLDist ;
                    REINFORCE.QR(iter) = QR ;
                    REINFORCE.QP(iter) = QP ;
                case 'REINFORCE2'
                    REINFORCE2.KLDist = KLDist ;
                    REINFORCE2.QR(iter) = QR ;
                    REINFORCE2.QP(iter) = QP ;
            end

            if (iter > 1) & (KLDist(iter) > KLDist(iter-1))
                a = 1; 
            end

            % Figure
%             figure(1),subplot(221),hold on;
%             h = stem(xx,nbinpdf(xx,QR,1-QP),'gx') ;
%             axis([xxmin,xxmax,0,yymax])
% 
%             figure(1),subplot(222),
%             plot(1:iter,KLDist,'r-'), xlim([1,MaxIter])
% 
%             pause(0.05)
        end
    end
    %%%%%%     Save
    if IsSave
        filename = ['NBNEWTR',num2str(PostR),'TP',num2str(PostP),'Adam',num2str(IsAdam),'LR',num2str(LR)] ;
        filename(filename=='.') = [] ;
        save(filename,'GO','REINFORCE','xx','PostPDF','xxmin','xxmax','yymax','PostR','PostP','IsAdam','LR')   
    end
end
    
