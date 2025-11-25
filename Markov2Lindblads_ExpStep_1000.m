function [plotSpan, RhoStore, dNs] = Markov2Lindblads_ExpStep_1000( ...
    RhoIn, Xhat, Phat, TSpan, hbar, mu, beta3, beta4, lambda, Hb, displayFlag)
% 1000-step fixed-time exponential integrator:
% Uses exactly 1000 uniform steps between TSpan(1) and TSpan(end).
% dt0 is ignored; time step is (TSpan(end) - TSpan(1))/1000.

tic

% ---- setup -------------------------------------------------------------
t  = TSpan(1);
tF = TSpan(end);
N  = size(RhoIn,1);
N2 = N*N;

% number of steps and fixed dt
nSteps = 1000;
if tF == t
    dt = 0;
else
    dt = (tF - t)/nSteps;
end

% Operators (sparse, Hermitianized defensively)
X = sparse(Xhat); X = 0.5*(X+X');
P = sparse(Phat); P = 0.5*(P+P');

% Building blocks
X2 = X'*X;                  % for dissipator
P2 = P'*P;

% Polynomial potential in X
Xpow2 = X*X;
Xpow3 = Xpow2*X;
Xpow4 = Xpow3*X;
Vchi = -(mu^2/2)*Xpow2+(2*beta3*mu/3)*Xpow3+((beta4^2-beta3^2)/4)*Xpow4 ;

% Hamiltonian split (time scaling applied later)
H1 = (0.5/Hb)*P2;
H2 = (1.0/Hb)*Vchi;

% Dissipation prefactors
cP  = (Hb/(2*pi));
cP2 = cP^2;

% ---- hoisted superoperators -------------------------------------------
I    = speye(N);
vecI = reshape(I, N2, 1);           % for fast trace without reshape

% Commutator superoperators: -i/hbar * (I⊗H - H^T⊗I)
LH1 = (-1i/hbar) * (kron(I, H1) - kron(H1.', I));
LH2 = (-1i/hbar) * (kron(I, H2) - kron(H2.', I));

% Lindblad dissipator superoperators (time-independent shape)
DX  = dissipator_super(X, X2, I);   % vec(D_X[rho]) = DX * vec(rho)
DP  = dissipator_super(P, P2, I);

% Vectorize initial state and normalize trace
y = RhoIn(:);
tr = real(vecI' * y); 
if tr ~= 0
    y = y / tr;
end

% ---- storage (store every step) ---------------------------------------
nSaves   = nSteps + 1;              % include initial state
plotSpan = zeros(1,nSaves);
dNs      = zeros(1,nSaves);
RhoStore = zeros(N, N, nSaves, 'like', RhoIn);

% store initial condition
storeCount = 1;
plotSpan(storeCount) = t;
RhoStore(:,:,storeCount) = full(reshape(y, N, N));
dNs(storeCount) = NaN;              % will overwrite later

% ---- main loop: 1000 fixed steps --------------------------------------
for k = 1:nSteps
    % midpoint for time-dependent coefficients
    tm = t + 0.5*dt;
    a2 = exp(3*tm);
    a1 = 1/a2;

    % Liouvillian as linear combo of prebuilt blocks
    % L = a1*LH1 + a2*LH2 + GGamma*DX + cP2*DP
    GGamma = (131*pi*lambda^2 * (a2*a2)) / (512*mu^5);
    L = a1*LH1 + a2*LH2 + GGamma*DX + cP2*DP;

    % exponential step
    y = expmv(L, y, dt);

    % advance time
    t = t + dt;

    % trace re-normalize using vec form (no reshape)
    tr = real(vecI' * y);
    if tr ~= 0
        y = y / tr;
    end

    % store
    storeCount        = storeCount + 1;
    plotSpan(storeCount) = t;
    dNs(storeCount)      = dt;

    R = reshape(y, N, N);
    RhoStore(:,:,storeCount) = full(R);

    if displayFlag
        phi    = real(trace(R*X));
        varPhi = real(trace(R*Xpow2)) - phi^2;
        purity = real(trace(R*R));
        fprintf(['λ=%+6.3f  Hb=%+6.3f  φ=%+6.3f  Var(φ)=%.3f  Purity=%.3f  ', ...
                 'N=%.3g  dN=%1.1e  time=%.3g\n'], ...
                 lambda, Hb, phi, varPhi, purity, t, dt, toc);
    end
end

% ---- trim / fix dNs ----------------------------------------------------
plotSpan = plotSpan(1:storeCount);
RhoStore = RhoStore(:,:,1:storeCount);
dNs      = dNs(1:storeCount);

% replace initial NaN step with the uniform dt (if any steps were taken)
if numel(dNs) >= 2 && isnan(dNs(1))
    dNs(1) = dNs(2);
end

end % function

% ---- helpers -----------------------------------------------------------
function D = dissipator_super(L, L2, I)
% D_L[ρ] = LρL† - 1/2(L†L ρ + ρ L†L)
% vec(D_L[ρ]) = (L* ⊗ L) vec(ρ) - 1/2( I ⊗ L†L + (L†L)^T ⊗ I ) vec(ρ)
Lt    = L.';        %#ok<NASGU>  % kept for clarity
Lc    = conj(L);
LdagL = L' * L;
D = kron(Lc, L) - 0.5*( kron(I, LdagL) + kron(LdagL.', I) );
end
