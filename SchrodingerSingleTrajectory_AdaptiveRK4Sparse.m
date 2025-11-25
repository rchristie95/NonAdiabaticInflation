function [plotSpan, PsiStore, RhoStore] = SchrodingerSingleTrajectory_AdaptiveRK4Sparse( PsiIn, Xhat, Phat, TSpan, hbar, mu, beta3, beta4, Hb)
% SchrodingerSingleTrajectory_AdaptiveRK4
%      Adaptive‑step RK4 integrator for
%      dψ/dN = -(i/ħ) H(N) ψ ,   H(N)=½ P̂² e^{-3N}+Vχ e^{+3N}
%
% INPUTS  (same as old routine, plus:)
%   tol   – desired local error tolerance (e.g. 1e‑7)
%
% OUTPUTS identical in shape to the fixed‑step version.

%% ----- adaptive parameters -------------------------------------------
growExp   = 1/5;      shrinkExp = 1/5;   % RK4 local order = 5
sfGrow    = 0.9;      sfShrink  = 0.9;
maxGrow   = 2.0;      minShrink = 0.1;
dtMax     = 1e-2;
tol =   5e-4;
%% ---------------------------------------------------------------------

% ---------- initialisation --------------------------------------------
N0     = TSpan(1);          Nend  = TSpan(end);
t_track=N0;
dt = 1e-6;
StoreChunk=(Nend-N0)/1000;
psi    = PsiIn ./ norm(PsiIn);
Ndim   = size(psi,1);       Id = eye(Ndim);

plotSpan(1) = N0;
PsiStore(:,1)      = psi;
RhoStore(:,:,1)    = psi*psi';
snap               = 1;

Id   = speye(Ndim);                 % sparse identity
Xhat    = sparse(Xhat);                % ensure sparse operators
Phat    = sparse(Phat);
Phat2=Phat*Phat';



% static part of the potential
Vchi = -(mu^2/2)*(Xhat*Xhat')+(2*beta3*mu/3)*Xhat^3+((beta4^2-beta3^2)/4)*(Xhat*Xhat')^2 ;

% main adaptive loop
Ncur     = N0;        accepted = 0;   attempted = 0;
while (Ncur < Nend) && (snap < 1000)
    attempted = attempted + 1;

    if Ncur + dt > Nend
        dt = Nend - Ncur;             % clamp final step
    end

    % ----------- one full RK4 vs. two half‑steps ----------------------
    psi_full  = RK4_step(psi, Ncur,       dt,   Xhat,Phat2,Vchi,Hb,hbar);
    psi_half  = RK4_step(psi, Ncur,       dt/2, Xhat,Phat2,Vchi,Hb,hbar);
    psi_2half = RK4_step(psi_half, Ncur+dt/2, dt/2, Xhat,Phat2,Vchi,Hb,hbar);

    err = norm(psi_full - psi_2half);

    if err < tol || dt < 1e-14      % -------- accept step ------------
        psi    = psi_2half / norm(psi_2half);
        Ncur   = Ncur + dt;
        accepted = accepted + 1;

        % propose next dt
        if err==0, fac = maxGrow; else
            fac = sfGrow * (tol/err)^growExp;
        end
        dt = min(dtMax, dt * min(maxGrow,fac));

        if abs(Ncur-t_track)>=StoreChunk
            t_track=Ncur;
            snap = snap + 1;
            plotSpan(snap)       = Ncur;
            PsiStore(:,snap)     = psi;
            RhoStore(:,:,snap)   = psi*psi';

            phi    = real(psi' * Xhat * psi);
            varPhi = real(psi' * (Xhat^2) * psi) - phi^2;
            fprintf('AdaptiveRK4: φ=%+.3f  Var(φ)=%.3f  N=%+.3f  dN=%1.1e  acc/att=%.3f,   mu/Hb=%.3f\n', phi,varPhi,Ncur,dt,accepted/attempted,mu/Hb);
        end
    else                          % -------- reject step --------------
        dt = dt * max(minShrink, sfShrink * (tol/err)^shrinkExp);
    end
end

% % trim unused cells
% plotSpan = plotSpan(1:snap);
% PsiStore = PsiStore(:,1:snap);
% RhoStore = RhoStore(:,:,1:snap);
end  % ========================= END main ===============================


%% ---------------------------------------------------------------------
%% Runge–Kutta 4 helper
%% ---------------------------------------------------------------------
function psiNew = RK4_step(psi,N0,dt,Xhat,Phat2,Vchi,Hb,hbar)
    % Hamiltonians at t, mid, end
    H0  = Ham(N0);
    Hm  = Ham(N0+0.5*dt);
    H1  = Ham(N0+dt);

    k1  = -(1i/hbar) * H0 * psi;
    k2  = -(1i/hbar) * Hm * (psi + 0.5*dt*k1);
    k3  = -(1i/hbar) * Hm * (psi + 0.5*dt*k2);
    k4  = -(1i/hbar) * H1 * (psi + dt*k3);

    psiNew = psi + dt*(k1 + 2*k2 + 2*k3 + k4)/6;

    % -------- nested helper for Hamiltonian
    function H = Ham(N)
        H = (0.5*Phat2*exp(-3*N) + Vchi*exp(3*N)) / Hb;
    end
end
