function [tvals, PsiOut, RhoOut] = SSEDynamicsMarkov_PlotSpan2LindsSparse( PsiIn, hbar, mu, beta3, beta4, lambda, Hb, Xhat, Phat, PlotSpanLind,dNLind)

% ---------- setup ----------
N0      = PlotSpanLind(1);
bSize   = length(PsiIn);
PsiCurr = PsiIn / norm(PsiIn);

KnotN         = numel(PlotSpanLind);
tvals         = PlotSpanLind;
PsiOut        = zeros(bSize, KnotN);
RhoOut        = zeros(bSize, bSize, KnotN);
PsiOut(:,1)   = PsiCurr;
RhoOut(:,:,1) = PsiCurr*PsiCurr.';

Xhat    = sparse(Xhat);                % ensure sparse operators
Phat    = sparse(Phat);
Phat2=Phat*Phat';

Vchi = -(mu^2/2)*(Xhat*Xhat')+(2*beta3*mu/3)*Xhat^3+((beta4^2-beta3^2)/4)*(Xhat*Xhat')^2 ;

t     = N0;
count = 1;
tol   = 1e-10;

for s = 1:KnotN-1
    t_next = PlotSpanLind(s+1);
    dt_loc = max(dNLind(s), eps); % guard against zero or negative

    while t < t_next - tol
        dt  = min(dt_loc, t_next - t);    % never step past the knot
        dW1 = sqrt(dt) * randn;
        dW2 = sqrt(dt) * randn;

        % operators at t and t+dt
        [H0, L1_0, L2_0] = opPack(t,       Xhat, Phat,Phat2, lambda, mu,Vchi, Hb);
        [H1, L1_1, L2_1] = opPack(t + dt,  Xhat, Phat,Phat2, lambda, mu,Vchi, Hb);

        % one SRK(1) step with two noises
        PsiCurr = SRK_Step(PsiCurr, dt, dW1, dW2, H0, H1, L1_0, L1_1, L2_0, L2_1, bSize, hbar);

        t = t + dt;
    end

    % store exactly at the knot
    count = count + 1;
    PsiOut(:,count)   = PsiCurr;
    RhoOut(:,:,count) = PsiCurr*PsiCurr';
    phi = real(PsiCurr' * Xhat * PsiCurr);
    phiV = real(PsiCurr' * (Xhat^2) * PsiCurr) - phi^2;
    fprintf('Markov-SSE: φ=%+6.3f Var(φ)=%.3f N=%+.3f dN=%+.3g snap %d\n', phi, phiV, t, dt, count);

end


end % ==== main ====

%% ====================================================================
%%  Sub-functions
%% ====================================================================
function Psi_new = SRK_Step(Psi, dtLoc, dW1, dW2, H0, H1, L1_now, L1_next, L2_now, L2_next, bSz, hbarLoc)
    % Drift and stochastic terms at t
    DPsi0 = Drift(Psi, H0, L1_now, L2_now, bSz, hbarLoc);
    G1_0  = Stoch1(Psi, L1_now, bSz, hbarLoc);
    G2_0  = Stoch2(Psi, L2_now, bSz, hbarLoc);

    sqdt  = sqrt(dtLoc);
    eta1  = dW1 / sqdt;
    eta2  = dW2 / sqdt;
    etaSq1 = 0.5*(eta1^2 - 1)*sqdt;
    etaSq2 = 0.5*(eta2^2 - 1)*sqdt;

    % predictor states
    Psi1   = Psi + DPsi0*dtLoc;
    Psi2_1 = Psi + DPsi0*dtLoc + G1_0*etaSq1;
    Psi3_1 = Psi + DPsi0*dtLoc - G1_0*etaSq1;
    Psi2_2 = Psi + DPsi0*dtLoc + G2_0*etaSq2;
    Psi3_2 = Psi + DPsi0*dtLoc - G2_0*etaSq2;

    % SRK(1) update with two independent noises
    T1 = Psi + 0.5*(DPsi0 + Drift(Psi1, H1, L1_next, L2_next, bSz, hbarLoc))*dtLoc;
    T2 = G1_0*dW1 + G2_0*dW2;
    T3 = 0.5*sqdt * ( ...
         (Stoch1(Psi2_1, L1_next, bSz, hbarLoc) - Stoch1(Psi3_1, L1_next, bSz, hbarLoc)) + ...
         (Stoch2(Psi2_2, L2_next, bSz, hbarLoc) - Stoch2(Psi3_2, L2_next, bSz, hbarLoc)) );

    Psi_new = T1 + T2 + T3;
    Psi_new = Psi_new / norm(Psi_new); % renormalize
end

function out = Drift(PsiV, H, L1, L2, bSz, hbarV)
    % unitary + dissipators in SSE gauge used by Stoch1/Stoch2
    out = (-1i/hbarV) * (H*PsiV) ...
        + Drift1(PsiV, L1, bSz, hbarV) ...
        + Drift2(PsiV, L2, bSz, hbarV);
end

function out = Drift1(PsiV, L1, bSz, hbarV)
    EPsi = PsiV' * L1 * PsiV;                     % scalar
    out  = (1/hbarV) * ( ...
          (-0.5*(L1'*L1)) ...
        + (EPsi')*L1 ...
        - 0.5*(EPsi*EPsi')*eye(bSz) ) * PsiV;
end

function out = Drift2(PsiV, L2, ~, hbarV)
    out  = (1/hbarV) * (-0.5*(L2'*L2)) * PsiV;
end

function out = Stoch1(PsiV, L1, bSz, hbarV)
    out = ( L1 - (PsiV' * L1 * PsiV)*eye(bSz) ) * PsiV / sqrt(hbarV);
end

function out = Stoch2(PsiV, L2, ~, hbarV)
    out = (-1i * L2) * PsiV / sqrt(hbarV);
end

%% ====================================================================
%%  Operators
%% ====================================================================
function [H, L1, L2] = opPack(Ncur, Xhat, Phat,Phat2, lambda, mu,Vchi, Hb)
    Gamma = (131*pi*lambda^2 * exp(6*Ncur)) / (512*mu^5);
    H     = (0.5*(Phat2) * exp(-3*Ncur) + Vchi * exp(3*Ncur)) / Hb;
    L1    = sqrt(Gamma) * Xhat;
    L2    = (Hb/(2*pi)) * Phat; % constant in time here
end
