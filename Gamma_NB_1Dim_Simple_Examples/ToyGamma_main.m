% Gamma Toy Experiments
% 
% By Yulai Cong
% 2018.01.18
clear,clc,close all

LogReal = 1 ;
IsSave = 0 ;

IsAdam = 1 ;
Beta1Adam = 0.9 ;
Beta2Adam = 0.999 ;
epsilon = 1e-8 ;

MaxIter = 1000 ;
if IsAdam
    LR = 0.1 ;
else
    LR = 0.01 ;
end
VarSampleNum = 20 ;

%% Make Toy data
Chosen = 1 ;  % 2 or 5
%%%          1     2     3    4    5   6   7   8
AlphaSet = [1e-5, 0.01, 0.1, 0.5,  1, 10, 50, 100] ; 
xxmaxSet = [1e-7 , 5e-6,1e-5, 1e-4, 5, 30, 80, 150] ;
LRset   =  [0.3,   0.1,  0.1,   0, 0.01, 0,  0,  0.001] ;
if ~IsAdam
    LR = LRset(Chosen) ;
end
BetaSet  = 0.5 * ones(1,20) ;
xxminSet = [1e-9,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7] ;

PostAlpha = AlphaSet(Chosen) 
PostBeta = BetaSet(Chosen) 
xxmax = xxmaxSet(Chosen) ;
xxmin = xxminSet(Chosen) ;

xx = linspace(xxmin,xxmax,200) ;
if PostAlpha-1 < 0
    yymax = gampdf(xx(1),PostAlpha,1/PostBeta) ;
else
    yymax = gampdf((PostAlpha-1)/PostBeta,PostAlpha,1/PostBeta) ;
end    
    

PostPDF = gampdf(xx,PostAlpha,1/PostBeta) ;

% figure(1),subplot(221),
% plot(xx,PostPDF,'r-') ;
% axis([xxmin,xxmax,0, yymax])

%% Loss surface
if 0
    a = 1:0.5:120 ;
    b = 0.5:0.01:1.5 ;
    [A,B] = meshgrid(a,b) ;
    
    QAlpha = A ; 
    QBeta = B ;
    KLDist = (QAlpha-PostAlpha).*psi(QAlpha) - gammaln(QAlpha) + gammaln(PostAlpha)...
                        + PostAlpha.*log(QBeta./PostBeta) + (PostBeta-QBeta).*QAlpha./QBeta ;
    figure,mesh(A,B,KLDist)
end

%% BBVI + ADAM 
if 1
    for ChosenGrad = { 'GO' , 'G-Rep_stick' , 'G-Rep', 'RSVI-stick', 'RSVI' }
        ChosenGrad = ChosenGrad{1} ;
        % Initialize
        if LogReal
            QAlphaLog = 0 ;
            QBetaLog = 0 ;
            QAlpha = exp(QAlphaLog) ;
            QBeta = exp(QBetaLog) ;
        else
            QAlpha = 1 ;
            QBeta = 1 ;
        end
        KLDist = [] ;
        count = 0;

        m0Alpha = 0 ;    m0Beta = 0 ;
        v0Alpha = 0 ;    v0Beta = 0 ;
        % Iteration
        for iter = 1:MaxIter

            if iter > 1
                delete(h) ;
            end

            for iii = 1:VarSampleNum

                zhatsample = randg(QAlpha) ;
                zhatsample = max(1e-250, zhatsample) ;
                zsample = zhatsample / QBeta ;

                PfzPz = (PostAlpha-QAlpha)/zsample - (PostBeta-QBeta) ;
                
                switch ChosenGrad
                    case 'GO'
                        GGradAlpha = GO_Gamma_v1(zhatsample, QAlpha) / QBeta  * PfzPz ;
                        GGradBeta = -zsample / QBeta * PfzPz ;
                    case 'G-Rep_stick'
                        epsisample = (log(zsample)-psi(QAlpha)+log(QBeta)) / sqrt(psi(1,QAlpha)) ;

                        tmp = epsisample*psi(2,QAlpha)/2/sqrt(psi(1,QAlpha)) + psi(1,QAlpha) ;

                        halpha = zsample*tmp ;
                        mualpha = tmp + psi(2,QAlpha)/2/psi(1,QAlpha) ;
                        hbeta = -zsample/QBeta ;
                        mubeta = -1/QBeta ;

                        fz = (PostAlpha-QAlpha)*log(zsample) - (PostBeta-QBeta)*zsample + ...
                                PostAlpha*log(PostBeta) - gammaln(PostAlpha) - QAlpha*log(QBeta) + gammaln(QAlpha)  ;

                        PlogqPz = (QAlpha-1) / zsample - QBeta ;
                        PlogqPalpha = log(QBeta) - psi(QAlpha) + log(zsample) ;
                        PlogqPbeta = QAlpha/QBeta - zsample ;

                        GGradAlpha = PfzPz*halpha + fz*(PlogqPz*halpha + PlogqPalpha + mualpha) ;
                        GGradBeta = PfzPz*hbeta + fz*(PlogqPz*hbeta + PlogqPbeta + mubeta) ;
                    case 'G-Rep'
                        epsisample = (log(zsample)-psi(QAlpha)+log(QBeta)) / sqrt(psi(1,QAlpha)) ;

                        tmp = epsisample*psi(2,QAlpha)/2/sqrt(psi(1,QAlpha)) + psi(1,QAlpha) ;

                        halpha = zsample*tmp ;
                        mualpha = tmp + psi(2,QAlpha)/2/psi(1,QAlpha) ;
                        hbeta = -zsample/QBeta ;
                        mubeta = -1/QBeta ;
                        
                        fz1 = PostAlpha*log(PostBeta) - gammaln(PostAlpha) + (PostAlpha-1)*log(zsample) - PostBeta*zsample ;
                        PfzPz1 = (PostAlpha-1)/(zsample) - PostBeta ; 
                        
                        PHPalpha = 1 + (1-QAlpha) * psi(1,QAlpha) ;
                        PHPbeta = -1/QBeta ;
                        
                        PlogqPz = (QAlpha-1) / zsample - QBeta ;
                        PlogqPalpha = log(QBeta) - psi(QAlpha) + log(zsample) ;
                        PlogqPbeta = QAlpha/QBeta - zsample ;

                        GGradAlpha = PfzPz1*halpha + fz1*(PlogqPz*halpha + PlogqPalpha + mualpha) + PHPalpha ;
                        GGradBeta = PfzPz1*hbeta + fz1*(PlogqPz*hbeta + PlogqPbeta + mubeta) + PHPbeta ;
                        
                    case 'RSVI-stick'
                        if PostAlpha >= 1
                            epsisample = calc_epsilon(zhatsample, QAlpha); 

                            halpha = grad_reject_h(epsisample, QAlpha) ;
                            mualpha = gamma_grad_logr(epsisample, QAlpha) ;
                            PzPbeta = -zsample/QBeta ;

                            fz = (PostAlpha-QAlpha)*log(zsample) - (PostBeta-QBeta)*zsample + ...
                                    PostAlpha*log(PostBeta) - gammaln(PostAlpha) - QAlpha*log(QBeta) + gammaln(QAlpha)  ;

                            PlogqPz = (QAlpha-1) / zhatsample - 1 ;
                            PlogqPalpha = - psi(QAlpha) + log(zhatsample) ;

                            GGradAlpha = PfzPz / QBeta * halpha + fz*(PlogqPz*halpha + PlogqPalpha + mualpha) ;
                            GGradBeta = PfzPz*PzPbeta  ;
                        else
                            UN = 5 ;
                            z_jsample = randg(QAlpha+UN) ;
                            z_jsample = max(1e-250, z_jsample) ;

                            epsisample = calc_epsilon(z_jsample, QAlpha+UN);

                            U = rand(1,UN) ;
                            zhatsample = prod( U.^(1./(QAlpha+[0:1:UN-1])) )*z_jsample;

                            zsample = zhatsample / QBeta ;

                            PfzPz = (PostAlpha-QAlpha)/zsample - (PostBeta-QBeta) ;   
                            PfzPzj = PfzPz / QBeta * prod( U.^(1./(QAlpha+[1:1:UN]-1)) );
                            PfzPalpha = PfzPz * zsample * sum( -log(U)./(QAlpha+[1:1:UN]-1)./(QAlpha+[1:1:UN]-1) ) ;

                            halpha = grad_reject_h(epsisample, QAlpha+UN) ;
                            mualpha = gamma_grad_logr(epsisample, QAlpha+UN) ;     
                            PzPbeta = -zsample/QBeta ;

                            fz = (PostAlpha-QAlpha)*log(zsample) - (PostBeta-QBeta)*zsample + ...
                                    PostAlpha*log(PostBeta) - gammaln(PostAlpha) - QAlpha*log(QBeta) + gammaln(QAlpha)  ;

                            PlogqPz = (QAlpha+UN - 1) / z_jsample - 1 ;
                            PlogqPalpha = - psi(QAlpha+UN) + log(z_jsample) ;

                            GGradAlpha = PfzPzj*halpha + PfzPalpha + fz*(PlogqPz*halpha + PlogqPalpha + mualpha) ;
                            GGradBeta = PfzPz*PzPbeta  ;
                        end
                    case 'RSVI'
                        if PostAlpha >= 1
                            epsisample = calc_epsilon(zhatsample, QAlpha); 

                            halpha = grad_reject_h(epsisample, QAlpha) ;
                            mualpha = gamma_grad_logr(epsisample, QAlpha) ;
                            PzPbeta = -zsample/QBeta ;

                            fz1 = PostAlpha*log(PostBeta) - gammaln(PostAlpha) + (PostAlpha-1)*log(zsample) - PostBeta*zsample ;
                            PfzPz1 = (PostAlpha-1)/(zsample) - PostBeta  ; 

                            PHPalpha = 1 + (1-QAlpha) * psi(1,QAlpha) ;
                            PHPbeta = -1/QBeta ;

                            PlogqPz = (QAlpha-1) ./ zhatsample - 1 ;
                            PlogqPalpha = - psi(QAlpha) + log(zhatsample) ;

                            GGradAlpha = PfzPz1 ./ QBeta *halpha + fz1*(PlogqPz*halpha + PlogqPalpha + mualpha) + PHPalpha ;
                            GGradBeta = PfzPz1*PzPbeta + PHPbeta ;
                        else
                            UN = 5 ;
                            z_jsample = randg(QAlpha+UN) ;
                            z_jsample = max(1e-250, z_jsample) ;

                            epsisample = calc_epsilon(z_jsample, QAlpha+UN); 

                            U = rand(1,UN) ;
                            zhatsample = prod( U.^(1./(QAlpha+[0:1:UN-1])) )*z_jsample;

                            zsample = zhatsample / QBeta ;

                            PfzPz = (PostAlpha- 1 )/zsample - PostBeta ;    
                            PfzPzj = PfzPz / QBeta * prod( U.^(1./(QAlpha+[1:1:UN]-1)) );
                            PfzPalpha = PfzPz * zsample * sum( -log(U)./(QAlpha+[1:1:UN]-1)./(QAlpha+[1:1:UN]-1) ) ;

                            halpha = grad_reject_h(epsisample, QAlpha+UN) ;
                            mualpha = gamma_grad_logr(epsisample, QAlpha+UN) ;
                            PzPbeta = -zsample/QBeta ;

                            fz1 = PostAlpha*log(PostBeta) - gammaln(PostAlpha) + (PostAlpha-1)*log(zsample) - PostBeta*zsample ;
                            PHPalpha = 1 + (1-QAlpha) * psi(1,QAlpha) ;
                            PHPbeta = -1/QBeta ;

                            PlogqPz = (QAlpha + UN - 1) / z_jsample - 1 ;
                            PlogqPalpha = - psi(QAlpha+UN) + log(z_jsample) ;

                            GGradAlpha = PfzPzj*halpha + PfzPalpha + fz1*(PlogqPz*halpha + PlogqPalpha + mualpha) + PHPalpha;
                            GGradBeta = PfzPz*PzPbeta  + PHPbeta ;
                        end                       
                        
                        if GGradAlpha > 1e3
                            a = 1 ;
                        end
                        
                    otherwise
                        error('Undefined Gradient!')
                end

                if LogReal
                    GradAlphaLog = GGradAlpha * QAlpha ; 
                    GradBetaLog = GGradBeta * QBeta ;            
                    GradCun(iii,1) = GradAlphaLog ;
                    GradCun(iii,2) = GradBetaLog ;
                else
                    GradCun(iii,1) = GGradAlpha ;
                    GradCun(iii,2) = GGradBeta ;
                end
            end

            switch ChosenGrad
                case 'GO'
                    GO.GradMean(iter,:) = mean(GradCun,1) ;
                    GO.GradVar(iter,:) = var(GradCun,[],1) ;
                case 'G-Rep_stick'
                    GRepStick.GradMean(iter,:) = mean(GradCun,1) ;
                    GRepStick.GradVar(iter,:) = var(GradCun,[],1) ;
                case 'G-Rep'
                    GRep.GradMean(iter,:) = mean(GradCun,1) ;
                    GRep.GradVar(iter,:) = var(GradCun,[],1) ;
                case 'RSVI-stick'
                    RSVIStick.GradMean(iter,:) = mean(GradCun,1) ;
                    RSVIStick.GradVar(iter,:) = var(GradCun,[],1) ;
                case 'RSVI'
                    RSVI.GradMean(iter,:) = mean(GradCun,1) ;
                    RSVI.GradVar(iter,:) = var(GradCun,[],1) ;
            end


            if LogReal
                GradAlphaLog = GGradAlpha * QAlpha ; 
                GradBetaLog = GGradBeta * QBeta ;


                % ADAM or PureGradient
                if IsAdam
                    m0Alpha = Beta1Adam*m0Alpha + (1-Beta1Adam)*GradAlphaLog ;
                    v0Alpha = Beta2Adam*v0Alpha + (1-Beta2Adam)*GradAlphaLog^2 ;
                    m0Beta = Beta1Adam*m0Beta + (1-Beta1Adam)*GradBetaLog ;
                    v0Beta = Beta2Adam*v0Beta + (1-Beta2Adam)*GradBetaLog^2 ;

                    m0AlphaHat = m0Alpha/(1-Beta1Adam^iter) ;
                    v0AlphaHat = v0Alpha/(1-Beta2Adam^iter) ;
                    m0BetaHat = m0Beta/(1-Beta1Adam^iter) ;
                    v0BetaHat = v0Beta/(1-Beta2Adam^iter) ;

                    QAlphaLog = QAlphaLog + LR * m0AlphaHat / (sqrt(v0AlphaHat)+epsilon)  ;
                    QBetaLog = QBetaLog + LR * m0BetaHat / (sqrt(v0BetaHat)+epsilon)  ;
                else
                    QAlphaLog = QAlphaLog + LR * GradAlphaLog ;
                    QBetaLog = QBetaLog + LR * GradBetaLog ;
                end
                QAlpha = exp(QAlphaLog) ;
                QBeta = exp(QBetaLog) ;

            else
                % ADAM or PureGradient
                if IsAdam
                    m0Alpha = Beta1Adam*m0Alpha + (1-Beta1Adam)*GGradAlpha ;
                    v0Alpha = Beta2Adam*v0Alpha + (1-Beta2Adam)*GGradAlpha^2 ;
                    m0Beta = Beta1Adam*m0Beta + (1-Beta1Adam)*GGradBeta ;
                    v0Beta = Beta2Adam*v0Beta + (1-Beta2Adam)*GGradBeta^2 ;

                    m0AlphaHat = m0Alpha/(1-Beta1Adam^iter) ;
                    v0AlphaHat = v0Alpha/(1-Beta2Adam^iter) ;
                    m0BetaHat = m0Beta/(1-Beta1Adam^iter) ;
                    v0BetaHat = v0Beta/(1-Beta2Adam^iter) ;

                    QAlpha = softplus_k(QAlpha + LR * m0AlphaHat / (sqrt(v0AlphaHat)+epsilon) ,  1e4) ;
                    QBeta = softplus_k(QBeta + LR * m0BetaHat / (sqrt(v0BetaHat)+epsilon) ,  1e4) ;
                else
                    QAlpha = softplus_k(QAlpha + LR * GGradAlpha,  1e4) ;
                    QBeta = softplus_k(QBeta + LR * GGradBeta, 1e4) ;
                end
            end

            if isnan(QAlpha) | isnan(QBeta)
                a = 1 ;
            end

            % Evaluate with iterations
            KLDist(iter) = (QAlpha-PostAlpha)*psi(QAlpha) - gammaln(QAlpha) + gammaln(PostAlpha)...
                        + PostAlpha*log(QBeta/PostBeta) + (PostBeta-QBeta)*QAlpha/QBeta ;
            fprintf('Iter = %d, KLDist = %d, QA = %d (%d), QB = %d (%d) \n', iter,KLDist(iter), QAlpha, PostAlpha,QBeta,PostBeta)
            
            switch ChosenGrad
                case 'GO'
                    GO.KLDist = KLDist ;
                    GO.QAlpha(iter) = QAlpha ;
                    GO.QBeta(iter) = QBeta ;
                case 'G-Rep_stick'
                    GRepStick.KLDist = KLDist ;
                    GRepStick.QAlpha(iter) = QAlpha ;
                    GRepStick.QBeta(iter) = QBeta ;
                case 'G-Rep'
                    GRep.KLDist = KLDist ;
                    GRep.QAlpha(iter) = QAlpha ;
                    GRep.QBeta(iter) = QBeta ;
                case 'RSVI-stick'
                    RSVIStick.KLDist = KLDist ;
                    RSVIStick.QAlpha(iter) = QAlpha ;
                    RSVIStick.QBeta(iter) = QBeta ;                  
                case 'RSVI'
                    RSVI.KLDist = KLDist ;
                    RSVI.QAlpha(iter) = QAlpha ;
                    RSVI.QBeta(iter) = QBeta ;      
            end

            if (iter > 1) & (KLDist(iter) > KLDist(iter-1))
                a = 1; 
            end

            % Figure
            figure(1),subplot(221),hold on;
            h = plot(xx,gampdf(xx,QAlpha,1/QBeta),'g-') ;
            axis([xxmin,xxmax,0, yymax])

            figure(1),subplot(222),
            plot(1:iter,KLDist,'r-'), xlim([1,MaxIter])

            pause(0.001)
        end
        
    end
    
    %%%%%%     Save
    if IsSave
        filename = ['GamNEWTAlpha',num2str(PostAlpha),'TBeta',num2str(PostBeta),'Adam',num2str(IsAdam),'LR',num2str(LR)] ;
        filename(filename=='.') = [] ;
        save(filename,'GO','GRepStick','GRep','RSVIStick','RSVI','xx','PostPDF','xxmin','xxmax','yymax','PostAlpha','PostBeta','IsAdam','LR')   
    end
    
end

 