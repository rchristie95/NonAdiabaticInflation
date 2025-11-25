



%% 

clc;
clearvars;
close all;
tStart = cputime;

%% Physical & Numerical Parameters
hbar      = 1;        % Reduced Planck constant

mu    = 0.5;      % Potential parameter: height scale
beta3=0.025;        % Potential parameter: separation between minima
beta4=0.13;



Hb        =0.5;     % Hubble scale parameter

lambda    =0.05;      % Coupling strength

% Time‐integration grid for SSE
N0        =-2;                       % Initial e-fold
Nend  = 1;

TSpan = linspace(N0, Nend,2);
targets = [-2, 0.0, 2];

% Basis sizes
bSize     =600;       % Fock basis dimension
Nx        = 200;      % Grid dimension for x/p representation

% Wigner video parameters
Xview     =8;        % Plot limits in phase space

%% Precompute Fock Operators
% Creation/annihilation operators
a     = zeros(bSize);
for n = 1 : bSize-1
    a(n,n+1) = sqrt(n);
end
adag  = a';
Xhat  = sqrt(hbar/2) * (adag + a);
Phat  = 1i * sqrt(hbar/2) * (adag - a);
Id    = eye(bSize);

[V,D] = eig(Xhat);
xvals  = diag(D);

%--- Numerical tolerance for "zero" ---
tol = 10*eps(max(abs(xvals)));   % loose-ish tolerance

%--- Indices of eigenvalues strictly > 0 (exclude ~0) ---
idx_pos = xvals > tol;

%--- Projector onto x>0 ---
ProjR = V(:,idx_pos) * V(:,idx_pos)';   % (bSize x bSize) Hermitian idempotent

%--- (Optional) enforce Hermiticity/idempotency tiny cleanup ---
ProjR = (ProjR + ProjR')/2;


%% Initial Instantaneous Hamiltonian & State
H0     = @(Ne) (0.5/Hb) * (Phat^2 * exp(-3*Ne) ...
    + (-(mu^2/2)*Xhat^2 + (2*mu*beta3/3)*Xhat^3 + (beta4^2-beta3^2)*Xhat^4/4) * exp(3*Ne));

% Build the Hamiltonian for the chosen e-fold N0
Hinit = H0(N0);

% --- small/medium matrix: use eig and sort -------------------------------
[Vec, E] = eig(full(Hinit));                 % E is diagonal
[~, idx] = min(diag(E));                 % position of lowest eigen-value
PsiIn = Vec(:, idx);
PsiIn = PsiIn / norm(PsiIn);             % normalise





%% Construct Hermite‐function Basis for x‐space

dX=sqrt(2*pi/(Nx*hbar));
% dX=3/100;

dP=2*pi/(dX*Nx/hbar);
for n=1:Nx
    x(n)=(-Nx/2+(n-1/2))*dX;
    p(n)=(-Nx/2+(n-1/2))*dP; %shift
end

HVector=zeros(Nx,Nx);
for n = 0:Nx-1
    HVector(:, n+1) = sqrt(dX / (2^n * factorial(n))) * (pi * hbar)^(-0.25).* exp(-x.^2 / (2 * hbar)) .* hermite_poly(n, x / sqrt(hbar));
end


% 
trace(ProjR*(PsiIn*PsiIn'))



%% --- ground states at requested e-folds ---
%% Three static Wigner pcolor+contour panels at N_e = -1.5, 0, 1.5 (no animation)
% 
% 
% Indices nearest to requested e-folds
for m = 1:3
    % Build the Hamiltonian for the chosen e-fold N0
    HOp = H0(targets(m));
    % --- small/medium matrix: use eig and sort -------------------------------
    [Vec, E] = eig(full(HOp));                 % E is diagonal
    [~, idx] = min(diag(E));                 % position of lowest eigen-value
    PsiImage = Vec(:, idx);
    ProjRGround(m)=abs((trace(ProjR*(PsiImage*PsiImage'))));
    PsiImage=PsiImage(1:Nx);
    PsiImage = HVector*PsiImage / norm(PsiImage);             % normalise
    WigImages(:,:,m)= PsiWigner(PsiImage, x, p, hbar);
end

% % Phase-space grid
[xMesh, pMesh] = meshgrid(x, p);
Xc=linspace(-Xview, Xview, Nx);
Pc=linspace(-Xview, Xview, Nx);
[xMesh1, pMesh1] = meshgrid(Xc, Pc);
% 
% Helper for Hamiltonian contour at a given frame index
Hgrid_at = @(k) ( exp(3*k) .* (-(mu^2/2).*xMesh1.^2 +(2*mu*beta3/3).*xMesh1.^3 +((beta4^2-beta3^2)/4).*xMesh1.^4) ...
                + 0.5 .* pMesh1.^2 .* exp(-3*k) ) / Hb;
% 
% % Common plotting settings
cax = [-0.05 0.15];
fmtText = @(Nval,Cval) sprintf('$N=%.2f,\\;\\langle\\hat\\theta_{\\phi^{+}}\\rangle=%.2f$', Nval, Cval);
% 
% % ---------- (a) N_e = -1.5 ----------
figA = figure('Position', [100 100 800 600]);  % wider figure for visibility
axA = axes(figA); hold(axA,'on');
hA = imagesc(axA, x, p, WigImages(:,:,1));
set(hA,'Interpolation','bilinear');   % or 'bicubic'
colormap(axA,'parula');
contour(axA, xMesh1, pMesh1, Hgrid_at(targets(1)), 25, 'LineColor', 'k');
xlim(axA,[-Xview Xview]); ylim(axA,[-Xview Xview]); axis(axA,'square'); box(axA,'on'); caxis(axA,cax);
xticks(linspace(-Xview,Xview,9))
colorbar(axA,'FontSize',18);
text(axA, 0.98, 0.98, fmtText(targets(1), ProjRGround(1)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(axA, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(axA,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(axA,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(axA, '(a)','Interpreter','latex','FontSize',30);
box on
exportgraphics(figA,'wigner_panel_a_Ne_-1p5.pdf','ContentType','vector');

% ---------- (b) N_e = 0 ----------
figB = figure('Position', [100 100 800 600]);  % wider figure for visibility
axB = axes(figB); hold(axB,'on');
hB = imagesc(axB, x, p, WigImages(:,:,2));
set(hB,'Interpolation','bilinear');   % or 'bicubic'
colormap(axB,'parula');
contour(axB, xMesh1, pMesh1, Hgrid_at(targets(2)), 25, 'LineColor', 'k');
xlim(axB,[-Xview Xview]); ylim(axB,[-Xview Xview]); axis(axB,'square'); box(axB,'on'); caxis(axB,cax);
xticks(linspace(-Xview,Xview,9))

colorbar(axB,'FontSize',18);
text(axB, 0.98, 0.98, fmtText(targets(2), ProjRGround(2)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(axB, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(axB,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(axB,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(axB, '(b)','Interpreter','latex','FontSize',30);
box on
exportgraphics(figB,'wigner_panel_b_Ne_0.pdf','ContentType','vector');

% ---------- (c) N_e = +1.5 ----------
figA = figure('Position', [100 100 800 600]);  % wider figure for visibility
axA = axes(figA); hold(axA,'on');
hC = imagesc(axA, x, p, WigImages(:,:,3));
set(hC,'Interpolation','bilinear');   % or 'bicubic'
colormap(axA,'parula');
contour(axA, xMesh1, pMesh1, Hgrid_at(targets(3)), 25, 'LineColor', 'k');
xlim(axA,[-Xview Xview]); ylim(axA,[-Xview Xview]); axis(axA,'square'); box(axA,'on'); caxis(axA,cax);
xticks(linspace(-Xview,Xview,9))

colorbar(axA,'FontSize',18);
text(axA, 0.98, 0.98, fmtText(targets(3), ProjRGround(3)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(axA, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(axA,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(axA,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(axA, '(c)','Interpreter','latex','FontSize',30);
box on
exportgraphics(figA,'wigner_panel_c_Ne_+0p75.pdf','ContentType','vector');



%% Single Reference Schrodinger


tic
[plotSpanHam,PsiSchrodinger , RhoSchrodinger]=SchrodingerSingleTrajectory_AdaptiveRK4Sparse( PsiIn, Xhat, Phat, TSpan, hbar,mu,beta3, beta4, Hb);
timeHam=toc

nFramesHam=length(plotSpanHam);


expectHSchrodinger = zeros(1,nFramesHam);
PuritySchrodinger  = zeros(1,nFramesHam);
NormSchrodinger = zeros(1,nFramesHam);
CtSchrodinger      = zeros(1,nFramesHam);
expXSchrodinger    = zeros(1,nFramesHam);
expPSchrodinger    = zeros(1,nFramesHam);
varXSchrodinger    = zeros(1,nFramesHam);
varPSchrodinger    = zeros(1,nFramesHam);
varXPSchrodinger    = zeros(1,nFramesHam);

PsiSchrodingerPos  = zeros(Nx,nFramesHam);
% RhoSchrodingerPos  = zeros(Nx,Nx,nFramesHam);

for k = 1 : nFramesHam
    NeCurr   = plotSpanHam(k);
    Hcurr    = H0(NeCurr);
    Rho        = RhoSchrodinger(:,:,k);

    expectHSchrodinger(k)  = real(trace(Hcurr * Rho));
    % Rho=Rho/trace(Rho);
    NormSchrodinger(k)  = real(trace(Rho));
    PuritySchrodinger(k)   = trace(Rho^2)/trace(Rho)^2;
    CtSchrodinger(k)       = abs(trace(ProjR * Rho*ProjR )) / NormSchrodinger(k);
    expXSchrodinger(k)     = real(trace(Xhat * Rho));
    expPSchrodinger(k)     = real(trace(Phat * Rho));
    varXSchrodinger(k)=real(trace(Rho*Xhat^2))-expXSchrodinger(k)^2;
    varPSchrodinger(k)=real(trace(Rho*Phat^2))-expPSchrodinger(k)^2;
    varXPSchrodinger(k)=0.5*real(trace(Rho*(Phat*Xhat+Xhat*Phat)))-expPSchrodinger(k)*expXSchrodinger(k);


    PsiSchrodingerPos(:,k) = HVector * PsiSchrodinger(1:Nx,k);
%     RhoSchrodingerPos(:,:,k) = HVector * RhoSchrodinger(:,:,k) * HVector';
end
% Schrodinger Video
clear flipbook
WigSchrodinger = PsiWigner(PsiSchrodingerPos, x, p, hbar);
% WigSSE = RhoWigner(RhoSchrodingerPos, x, p, hbar);

%% Panels Hamilton
targets = [-1, 0.0, 1];

[~, idxOrig] = min(abs(plotSpanHam- targets(1)), [], 2);   % idx(i) indexes p nearest to t(i)
[~, idxZero] = min(abs(plotSpanHam- targets(2)), [], 2);   % idx(i) indexes p nearest to t(i)
WigImagesHam(:,:,1)=WigSchrodinger(:,:, idxOrig);
WigImagesHam(:,:,2)=WigSchrodinger(:,:, idxZero);
WigImagesHam(:,:,3)=WigSchrodinger(:,:, end);


% ---------- (a) N_e = -0.75 ----------
figA = figure('Position', [100 100 800 600]);  % wider figure for visibility
axA = axes(figA); hold(axA,'on');
hA = imagesc(axA, x, p, WigImagesHam(:,:,1));
set(hA,'Interpolation','bilinear');   % or 'bicubic'
colormap(axA,'parula');
contour(axA, xMesh1, pMesh1, Hgrid_at(targets(1)), 25, 'LineColor', 'k');
xlim(axA,[-Xview Xview]); ylim(axA,[-Xview Xview]); axis(axA,'square'); box(axA,'on'); caxis(axA,cax);
xticks(linspace(-Xview,Xview,9))
colorbar(axA,'FontSize',18);
expLine = plot(expXSchrodinger(1:idxOrig), expPSchrodinger(1:idxOrig), 'w-', 'LineWidth', 1); % White line with specified width
text(axA, 0.98, 0.98, fmtText(targets(1), CtSchrodinger(idxOrig)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(axA, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(axA,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(axA,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(axA, '(a)','Interpreter','latex','FontSize',30);
box on
exportgraphics(figA,'Hamwigner_panel_a_Ne_-0p75.pdf','ContentType','vector');

% ---------- (b) N_e = 0 ----------
figB = figure('Position', [100 100 800 600]);  % wider figure for visibility
axB = axes(figB); hold(axB,'on');
hB = imagesc(axB, x, p, WigImagesHam(:,:,2));
set(hB,'Interpolation','bilinear');   % or 'bicubic'
colormap(axB,'parula');
contour(axB, xMesh1, pMesh1, Hgrid_at(targets(2)), 25, 'LineColor', 'k');
xlim(axB,[-Xview Xview]); ylim(axB,[-Xview Xview]); axis(axB,'square'); box(axB,'on'); caxis(axB,cax);
xticks(linspace(-Xview,Xview,9))
colorbar(axB,'FontSize',18);
expLine = plot(expXSchrodinger(1:idxZero), expPSchrodinger(1:idxZero), 'w-', 'LineWidth', 1); % White line with specified width
text(axB, 0.98, 0.98, fmtText(targets(2), CtSchrodinger(idxZero)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(axB, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(axB,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(axB,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(axB, '(b)','Interpreter','latex','FontSize',30);
box on
exportgraphics(figB,'Hamwigner_panel_b_Ne_0.pdf','ContentType','vector');

% ---------- (c) N_e = +1.5 ----------
figA = figure('Position', [100 100 800 600]);  % wider figure for visibility
axA = axes(figA); hold(axA,'on');
hC = imagesc(axA, x, p, WigImagesHam(:,:,3));
set(hC,'Interpolation','bilinear');   % or 'bicubic'
colormap(axA,'parula');
contour(axA, xMesh1, pMesh1, Hgrid_at(targets(3)), 25, 'LineColor', 'k');
xlim(axA,[-Xview Xview]); ylim(axA,[-Xview Xview]); axis(axA,'square'); box(axA,'on'); caxis(axA,cax);
xticks(linspace(-Xview,Xview,9))
colorbar(axA,'FontSize',18);
expLine = plot(expXSchrodinger(1:end), expPSchrodinger(1:end), 'w-', 'LineWidth', 1); % White line with specified width
text(axA, 0.98, 0.98, fmtText(targets(3),CtSchrodinger(end)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(axA, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(axA,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(axA,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(axA, '(c)','Interpreter','latex','FontSize',30);
box on
exportgraphics(figA,'Hamwigner_panel_c_Ne_+0p75.pdf','ContentType','vector');


%% Hamiltonian Video
Xc=linspace(-Xview, Xview, Nx);
Pc=linspace(-Xview, Xview, Nx);
% H = @(Z,n) (6*exp(3*plotSpanHam(n))*sqrt((0.5*c2*Z(1)^2+0.5*Z(2)^2*exp(-6*plotSpanHam(n)))/3));
Hinit = @(Z,n) (exp(3*plotSpanHam(n))*(-(mu^2/2)*Z(1)^2+(2*mu*beta3/3)*Z(1)^3+((beta4^2-beta3^2)/4)*Z(1)^4)+0.5*Z(2)^2*exp(-3*plotSpanHam(n)))/Hb;

% Define your meshgrid
[xMesh, pMesh] = meshgrid(x, p); % Adjust the limits and resolution as needed
[xMesh1, pMesh1] = meshgrid(Xc, Pc); % Adjust the limits and resolution as needed

% Compute initial contour data
Z_initial = arrayfun(@(x, y) Hinit([x; y], 1), xMesh, pMesh);

% Define aspect ratio for the figure
aspectRatio = [6, 6];

% Create a figure with specified position and aspect ratio
figureHandle = figure('Position', [100, 100, 800, 800 * aspectRatio(2) / aspectRatio(1)]);

% Initialize pcolor plot with the first set of data
f = pcolor(xMesh, pMesh, WigSchrodinger(:,:,1));
xlim([-Xview Xview]);
ylim([-Xview Xview]);
shading interp
colormap("parula")  % Changed colormap to 'parula' for perceptually uniform colors

% Set color axis limits
clim([-0.05 0.15])  % Ensure this range covers both oscillator and inverted oscillator regimes

% Add a colorbar with increased font size
colorbar('FontSize', 14);

title(sprintf(...
    'Hamiltonian Evolution, $\\tilde \\mu = %.2f$',mu/Hb), ...
    'FontSize', 20, 'Interpreter', 'latex' );

xlabel('$\phi$', 'FontSize', 25, 'Interpreter', 'latex');
ylabel('$\pi_{\phi}$', 'FontSize', 25, 'Interpreter', 'latex');
xticks(linspace(-Xview,Xview,9))

% Customize axis properties for better aesthetics
set(gca, 'FontSize', 16, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
axis square  % Use square axes for equal aspect ratio

% Hold the current plot to allow multiple graphics objects
hold on;

% Initialize an empty line plot for experimental data
expLine = plot(NaN, NaN, 'w-', 'LineWidth', 1); % White line with specified width

% Initialize contour plot with initial Z data
[~, contPlot] = contour(xMesh1, pMesh1, Z_initial, 25, 'LineColor', "k"); % 100 contour levels in black

% Initialize the time display
timeText = text(0.95, 0.95, '', ...
    'Units', 'normalized', ...          % Position relative to axes (0 to 1)
    'HorizontalAlignment', 'right', ... % Align text to the right
    'VerticalAlignment', 'top', ...     % Align text to the top
    'FontSize', 25, ...                  % Set desired font size
    'FontWeight', 'bold', ...            % Make text bold for visibility
    'Color', 'w');                       % Set text color to black
uistack(timeText, 'top');                 % Ensure text is on top

% Main simulation loop
for n = 1:length(WigSchrodinger(1,1,:))
    % Retrieve the current simulation time from plotSpanHam
    t = plotSpanHam(n);

    % Update the pcolor plot with new data for the current time step
    set(f, 'CData', WigSchrodinger(:,:,n));

    % Compute contour data based on function H
    CdatCont = arrayfun(@(x, y) Hinit([x; y], n), xMesh1, pMesh1);

    % ---------------------- Dynamic Contour Update ----------------------

    % Delete existing contour plot to prevent overlap
    delete(contPlot);

    % Recompute and plot new contours with updated data
    [~, contPlot] = contour(xMesh1, pMesh1, CdatCont, 25, 'LineColor', "k");

    % Optionally, adjust contour levels here if needed
    % For example, to specify levels:
    % levels = linspace(min(CdatCont(:)), max(CdatCont(:)), 100);
    % [~, contPlot] = contour(xMesh, pMesh, CdatCont, levels, 'LineColor', "k");

    % ---------------------- End of Dynamic Contour Update -------------------

    % Update the experimental line plot with new data point
    xData = get(expLine, 'XData');
    yData = get(expLine, 'YData');
    set(expLine, 'XData', [xData, expXSchrodinger(n)], 'YData', [yData, expPSchrodinger(n)]);

    % ---------------------- Update Time Display ----------------------

    % Update the text object with the current time, formatted to two decimal places
    set(timeText, 'Interpreter', 'latex', ...
        'String', sprintf('$N = $ %.2f, $\\langle \\hat\\theta_{\\phi^{+}}\\rangle = $ %.2f', t, CtSchrodinger(n)));


    % Ensure the time text stays on top of other plot elements
    uistack(timeText, 'top');

    % ---------------------- End of Time Display Update -------------------

    % Refresh the plot to reflect updates
    drawnow;

    % Optional: Capture the frame for a flipbook or video
    flipbook(n) = getframe(gcf);
end

% Save video
writerObj = VideoWriter(sprintf('WignerSchrodingerVideo_Hubble_%.2g',Hb));
writerObj.FrameRate = nFramesHam/30;
open(writerObj);
writeVideo(writerObj, flipbook);
close(writerObj);

close all;

% Common plotting settings
cax = [-0.05 0.15];
fmtText = @(Nval,Cval) sprintf('$N=%.2f,\\;\\langle\\hat\\theta_{\\phi^{+}}\\rangle=%.2f$', Nval, Cval);


%% instanteneous trajectory
plotSpanInst=plotSpanHam;
for k = 1:nFramesHam
    % Build the Hamiltonian for the chosen e-fold N0
    NeCurr   = plotSpanHam(k);
    Hcurr    = H0(NeCurr);
        % --- small/medium matrix: use eig and sort -------------------------------
    [Vec, E] = eig(full(Hcurr));                 % E is diagonal
    [~, idx] = min(diag(E));                 % position of lowest eigen-value
    PsiInst(:,k)=Vec(:, idx);
    RhoInst(:,:,k)= PsiInst(:,k)*PsiInst(:,k)';
    Rho=RhoInst(:,:,k);

    expectHInst(k)  = real(trace(Hcurr * Rho));
    % Rho=Rho/trace(Rho);
    NormInst(k)  = real(trace(Rho));
    PurityInst(k)   = trace(Rho^2)/trace(Rho)^2;
    CtInst(k)       = abs(trace(ProjR * Rho*ProjR )) / NormInst(k);
    expXInst(k)     = real(trace(Xhat * Rho));
    expPInst(k)     = real(trace(Phat * Rho));
    varXInst(k)=real(trace(Rho*Xhat^2))-expXInst(k)^2;
    varPInst(k)=real(trace(Rho*Phat^2))-expPInst(k)^2;
    varXPInst(k)=0.5*real(trace(Rho*(Phat*Xhat+Xhat*Phat)))-expPInst(k)*expXInst(k);


    PsiInstPos(:,k) = HVector * PsiInst(1:Nx,k);

end

 %% InstVideo

clear flipbook

WigInst = PsiWigner(PsiInstPos, x, p, hbar);

Xc=linspace(-Xview, Xview, Nx);
Pc=linspace(-Xview, Xview, Nx);
% H = @(Z,n) (6*exp(3*plotSpan(n))*sqrt((0.5*c2*Z(1)^2+0.5*Z(2)^2*exp(-6*plotSpan(n)))/3));
Hinit = @(Z,n) (exp(3*plotSpanInst(n))*(-(mu^2/2)*Z(1)^2+(2*mu*beta3/3)*Z(1)^3+((beta4^2-beta3^2)/4)*Z(1)^4)+0.5*Z(2)^2*exp(-3*plotSpanInst(n)))/Hb;

% Define your meshgrid
[xMesh, pMesh] = meshgrid(x, p); % Adjust the limits and resolution as needed
[xMesh1, pMesh1] = meshgrid(Xc, Pc); % Adjust the limits and resolution as needed

% Compute initial contour data
Z_initial = arrayfun(@(x, y) Hinit([x; y], 1), xMesh, pMesh);

% Define aspect ratio for the figure
aspectRatio = [6, 6];

% Create a figure with specified position and aspect ratio
figureHandle = figure('Position', [100, 100, 800, 800 * aspectRatio(2) / aspectRatio(1)]);

% Initialize pcolor plot with the first set of data
f = pcolor(xMesh, pMesh, WigInst(:,:,1));
xlim([-Xview Xview]);
ylim([-Xview Xview]);
xticks(linspace(-Xview,Xview,9))

shading interp
colormap("parula")  % Changed colormap to 'parula' for perceptually uniform colors

% Set color axis limits
clim([-0.05 0.15])  % Ensure this range covers both oscillator and inverted oscillator regimes

% Add a colorbar with increased font size
colorbar('FontSize', 14);

title( sprintf(...
    'Instantaneous Ground State'), ...
    'FontSize', 20, 'Interpreter', 'latex' );

xlabel('$\phi$', 'FontSize', 25, 'Interpreter', 'latex');
ylabel('$\pi_{\phi}$', 'FontSize', 25, 'Interpreter', 'latex');

% Customize axis properties for better aesthetics
set(gca, 'FontSize', 16, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
axis square  % Use square axes for equal aspect ratio

% Hold the current plot to allow multiple graphics objects
hold on;

% Initialize an empty line plot for experimental data
expLine = plot(NaN, NaN, 'w-', 'LineWidth', 1); % White line with specified width

% Initialize contour plot with initial Z data
[~, contPlot] = contour(xMesh1, pMesh1, Z_initial, 25, 'LineColor', "k"); % 100 contour levels in black

% Initialize the time display
timeText = text(0.95, 0.95, '', ...
    'Units', 'normalized', ...          % Position relative to axes (0 to 1)
    'HorizontalAlignment', 'right', ... % Align text to the right
    'VerticalAlignment', 'top', ...     % Align text to the top
    'FontSize', 25, ...                  % Set desired font size
    'FontWeight', 'bold', ...            % Make text bold for visibility
    'Color', 'w');                       % Set text color to black
uistack(timeText, 'top');                 % Ensure text is on top

% Main simulation loop
for n = 1:length(WigInst(1,1,:))
    % Retrieve the current simulation time from plotSpanInst
    t = plotSpanInst(n);

    % Update the pcolor plot with new data for the current time step
    set(f, 'CData', WigInst(:,:,n));

    % Compute contour data based on function H
    CdatCont = arrayfun(@(x, y) Hinit([x; y], n), xMesh1, pMesh1);

    % ---------------------- Dynamic Contour Update ----------------------

    % Delete existing contour plot to prevent overlap
    delete(contPlot);

    % Recompute and plot new contours with updated data
    [~, contPlot] = contour(xMesh1, pMesh1, CdatCont, 25, 'LineColor', "k");

    % Optionally, adjust contour levels here if needed
    % For example, to specify levels:
    % levels = linspace(min(CdatCont(:)), max(CdatCont(:)), 100);
    % [~, contPlot] = contour(xMesh, pMesh, CdatCont, levels, 'LineColor', "k");

    % ---------------------- End of Dynamic Contour Update -------------------

    % Update the experimental line plot with new data point
    xData = get(expLine, 'XData');
    yData = get(expLine, 'YData');
    set(expLine, 'XData', [xData, expXInst(n)], 'YData', [yData, expPInst(n)]);

    % ---------------------- Update Time Display ----------------------

    % Update the text object with the current time, formatted to two decimal places
    set(timeText, 'Interpreter', 'latex', ...
        'String', sprintf('$N = $ %.2f, $\\langle \\hat\\theta_{\\phi^{+}}\\rangle = $ %.2f', t, CtInst(n)));


    % Ensure the time text stays on top of other plot elements
    uistack(timeText, 'top');

    % ---------------------- End of Time Display Update -------------------

    % Refresh the plot to reflect updates
    drawnow;

    % Optional: Capture the frame for a flipbook or video
    flipbook(n) = getframe(gcf);
end

% Save video
writerObj = VideoWriter(sprintf('WignerGroundStateVideo'));
writerObj.FrameRate = nFramesHam/30;
open(writerObj);
writeVideo(writerObj, flipbook);
close(writerObj);

close all;


%% Single Lindblad

tic
% [plotSpanLind, RhoLind,dNsLind] = Markov2Lindblads_Adaptive_Sparse_Dopri (PsiIn*PsiIn', Xhat, Phat, TSpan, hbar, mu,alpha3, alpha4, lambda, Hb,true);
[plotSpanLind, RhoLind,dNsLind] = Markov2Lindblads_ExpStep_1000 (PsiIn*PsiIn', Xhat, Phat, TSpan, hbar, mu,beta3, beta4, lambda, Hb,true);

timeLind=toc
dNsLind=(dNsLind./200)./exp(abs(3*(plotSpanLind+0.5)));
% dNsLind=(dNsLind)./exp(abs(3*(plotSpanLind+0.5)));

nFramesLind=length(plotSpanLind);


expectHLind = zeros(1,nFramesLind);
PurityLind  = zeros(1,nFramesLind);
NormLind = zeros(1,nFramesLind);
CtLind      = zeros(1,nFramesLind);
expXLind    = zeros(1,nFramesLind);
expPLind    = zeros(1,nFramesLind);
varXLind    = zeros(1,nFramesLind);
varPLind    = zeros(1,nFramesLind);
varXPLind    = zeros(1,nFramesLind);

PsiLindPos  = zeros(Nx,nFramesLind);
RhoLindPos  = zeros(Nx,Nx,nFramesLind);

for k = 1 : nFramesLind
    NeCurr   = plotSpanLind(k);
    Hcurr    = H0(NeCurr);
    Rho        = RhoLind(:,:,k);

    expectHLind(k)  = real(trace(Hcurr * Rho));
    % Rho=Rho/trace(Rho);
    NormLind(k)  = real(trace(Rho));
    PurityLind(k)   = trace(Rho^2)/trace(Rho)^2;
    CtLind(k)       = abs(trace(ProjR * Rho*ProjR )) / NormLind(k);
    expXLind(k)     = real(trace(Xhat * Rho));
    expPLind(k)     = real(trace(Phat * Rho));
    varXLind(k)=real(trace(Rho*Xhat^2))-expXLind(k)^2;
    varPLind(k)=real(trace(Rho*Phat^2))-expPLind(k)^2;
    varXPLind(k)=0.5*real(trace(Rho*(Phat*Xhat+Xhat*Phat)))-expPLind(k)*expXLind(k);

    RhoLindPos(:,:,k) = HVector *Rho(1:Nx,1:Nx)* HVector';

end
dotCtLind = gradient(CtLind, plotSpanLind);



% expXLind=expXHam;
% expPLind=expPHam;

save('workspaceSingles2')

%% Single NMQSD

rng('shuffle');   % seed RNG based on current time

% eta=(randn(Nsteps+1,1));
tic
[plotSpanSSE,PsiSSE, RhoSSE] = SSEDynamicsMarkov_PlotSpan2LindsSparse(   PsiIn, hbar, mu, beta3, beta4, lambda, Hb, Xhat, Phat,plotSpanLind,dNsLind);

% [plotSpanSSE,PsiSSE, RhoSSE] = SSEDynamicsMarkov_Adaptive(   PsiIn, TSpan, hbar, Separation, Height, Linear, lambda, Hb, epsilon, Xhat, Phat);
% [plotSpanSSE,PsiSSE, RhoSSE] = SSEDynamicsMarkov_ESRK1_RSwM3(   PsiIn, TSpan, hbar, Separation, Height, Linear, lambda, Hb, epsilon, Xhat, Phat);
% [plotSpanSSE,PsiSSE, RhoSSE] = SSEDynamicsMarkov_Adaptive2(   PsiIn, TSpan, hbar, Separation, Height, Linear, lambda, Hb, Xhat, Phat);

timeSSE=toc
nFramesSSE=length(plotSpanSSE);


expectHSSE = zeros(1,nFramesSSE);
PuritySSE  = zeros(1,nFramesSSE);
NormSSE = zeros(1,nFramesSSE);
CtSSE      = zeros(1,nFramesSSE);
expXSSE    = zeros(1,nFramesSSE);
expPSSE    = zeros(1,nFramesSSE);
varXSSE    = zeros(1,nFramesSSE);
varPSSE    = zeros(1,nFramesSSE);
varXPSSE    = zeros(1,nFramesSSE);

PsiSSEPos  = zeros(Nx,nFramesSSE);
RhoSSEPos  = zeros(Nx,Nx,nFramesSSE);

for k = 1 : nFramesSSE


    NeCurr   = plotSpanSSE(k);
    Hcurr    = H0(NeCurr);
    % RhoSSE(:,:,k)        = PsiSSE(:,k)*PsiSSE(:,k)';
    Rho        = RhoSSE(:,:,k);

    expectHSSE(k)  = real(trace(Hcurr * Rho));
    % Rho=Rho/trace(Rho);
    NormSSE(k)  = real(trace(Rho));
    PuritySSE(k)   = trace(Rho^2)/trace(Rho)^2;
    CtSSE(k)       = abs(trace(ProjR * Rho*ProjR )) / NormSSE(k);
    expXSSE(k)     = real(trace(Xhat * Rho));
    expPSSE(k)     = real(trace(Phat * Rho));
    varXSSE(k)=real(trace(Rho*Xhat^2))-expXSSE(k)^2;
    varPSSE(k)=real(trace(Rho*Phat^2))-expPSSE(k)^2;
    varXPSSE(k)=0.5*real(trace(Rho*(Phat*Xhat+Xhat*Phat)))-expPSSE(k)*expXSSE(k);


    PsiSSEPos(:,k) = HVector * PsiSSE(1:Nx,k);


end
dotCtSSE = gradient(CtSSE, plotSpanSSE);




%%  ⟨X⟩ SSE subplot
% Export 4 separate panels with LaTeX labels and no titles

lw = 1.5; fs = 25;

% 1) <X>
f1 = figure('Position',[100 100 800 600]); ax1 = axes(f1); hold(ax1,'on');
plot(plotSpanSSE,  expXSSE,         'b-', 'LineWidth', lw);
% plot(plotSpanSSE,  expXSSE,         'm-', 'LineWidth', lw);
plot(plotSpanHam,  expXSchrodinger, 'r-', 'LineWidth', lw);
plot(plotSpanLind, expXLind,        'k--','LineWidth', lw);
xlabel('$N$','Interpreter','latex','FontSize',fs);
ylabel('$\langle\hat  \phi \rangle$','Interpreter','latex','FontSize',fs);
title( '(a)','Interpreter','latex','FontSize',30);
grid on;    % add grid lines
% legend({'SSE','Schr\"odinger','Lindblad'},'Interpreter','latex','Location','best','Box','off');
set(ax1,'TickLabelInterpreter','latex','FontSize',fs,'LineWidth',1.2); axis(ax1,'tight'); box(ax1,'on');
exportgraphics(f1, sprintf('expX_vsNe_lambda_%.2g_Hubble_%.2g.pdf',lambda,Hb), 'ContentType','vector');

% 2) <P>
f2 = figure('Position',[100 100 800 600]); ax2 = axes(f2); hold(ax2,'on');
plot(plotSpanSSE,  expPSSE,         'b-', 'LineWidth', lw);
% plot(plotSpanSSE,  expPSSE,         'm-', 'LineWidth', lw);
plot(plotSpanHam,  expPSchrodinger, 'r-', 'LineWidth', lw);
plot(plotSpanLind, expPLind,        'k--','LineWidth', lw);
xlabel(' $N$','Interpreter','latex','FontSize',fs);
ylabel('$\langle \hat \pi_{\phi} \rangle$','Interpreter','latex','FontSize',fs);
title( '(b)','Interpreter','latex','FontSize',30);
grid on;    % add grid lines
% legend({'SSE','Schr\"odinger','Lindblad'},'Interpreter','latex','Location','best','Box','off');
set(ax2,'TickLabelInterpreter','latex','FontSize',fs,'LineWidth',1.2); axis(ax2,'tight'); box(ax2,'on');
exportgraphics(f2, sprintf('expP_vsNe_lambda_%.2g_Hubble_%.2g.pdf',lambda,Hb), 'ContentType','vector');

% 3) <H>
f3 = figure('Position',[100 100 800 600]); ax3 = axes(f3); hold(ax3,'on');
plot(plotSpanSSE,  expectHSSE,         'b-', 'LineWidth', lw);
% plot(plotSpanSSE,  expectHSSE,         'm-', 'LineWidth', lw);
plot(plotSpanHam,  expectHSchrodinger, 'r-', 'LineWidth', lw);
plot(plotSpanLind, expectHLind,        'k--','LineWidth', lw);
xlabel(' $N$','Interpreter','latex','FontSize',fs);
ylabel('$\langle \hat K_S \rangle$','Interpreter','latex','FontSize',fs);
title( '(c)','Interpreter','latex','FontSize',30);
grid on;    % add grid lines
% legend({'SSE (True Vacuum)','SSE (False Vacuum)','Schr\"odinger','Lindblad'},'Interpreter','latex','Location','best','Box','on');
legend({'SSE','Schr\"odinger','Lindblad'},'Interpreter','latex','Location','best','Box','on');

set(ax3,'TickLabelInterpreter','latex','FontSize',fs,'LineWidth',1.2); axis(ax3,'tight'); box(ax3,'on');
exportgraphics(f3, sprintf('expH_vsNe_lambda_%.2g_Hubble_%.2g.pdf',lambda,Hb), 'ContentType','vector');



%% Create Wigner‐Function Video
clear flipbook

WigSSE = PsiWigner(PsiSSEPos, x, p, hbar);
% WigSSE = RhoWigner(RhoSSEPos, x, p, hbar);

Xc=linspace(-Xview, Xview, Nx);
Pc=linspace(-Xview, Xview, Nx);
Hinit = @(Z,n) (exp(3*plotSpanSSE(n))*(-(mu^2/2)*Z(1)^2+(2*mu*beta3/3)*Z(1)^3+((beta4^2-beta3^2)/4)*Z(1)^4)+0.5*Z(2)^2*exp(-3*plotSpanSSE(n)))/Hb;

% Define your meshgrid
[xMesh, pMesh] = meshgrid(x, p); % Adjust the limits and resolution as needed
[xMesh1, pMesh1] = meshgrid(Xc, Pc); % Adjust the limits and resolution as needed

% Compute initial contour data
Z_initial = arrayfun(@(x, y) Hinit([x; y], 1), xMesh, pMesh);

% Define aspect ratio for the figure
aspectRatio = [6, 6];

% Create a figure with specified position and aspect ratio
figureHandle = figure('Position', [100, 100, 800, 800 * aspectRatio(2) / aspectRatio(1)]);

% Initialize pcolor plot with the first set of data
f = pcolor(xMesh, pMesh, WigSSE(:,:,1));
xlim([-Xview Xview]);
ylim([-Xview Xview]);
xticks(linspace(-Xview,Xview,9))
shading interp
colormap("parula")  % Changed colormap to 'parula' for perceptually uniform colors

% Set color axis limits
clim([-0.05 0.15])  % Ensure this range covers both oscillator and inverted oscillator regimes

% Add a colorbar with increased font size
colorbar('FontSize', 14);

title( sprintf(...
    'SSE Evolution, $\\lambda = %.2f$, $\\tilde \\mu = %.2f$', ...
    lambda, mu/Hb), ...
    'FontSize', 20, 'Interpreter', 'latex' );
% title( sprintf(...
%     'SSE Evolution, $H = %.2f$', Hb), ...
%     'FontSize', 20, 'Interpreter', 'latex' );

xlabel('$\phi$', 'FontSize', 25, 'Interpreter', 'latex');
ylabel('$\pi_{\phi}$', 'FontSize', 25, 'Interpreter', 'latex');

% Customize axis properties for better aesthetics
set(gca, 'FontSize', 16, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
axis square  % Use square axes for equal aspect ratio

% Hold the current plot to allow multiple graphics objects
hold on;

% Initialize an empty line plot for experimental data
expLine = plot(NaN, NaN, 'w-', 'LineWidth', 1); % White line with specified width

% Initialize contour plot with initial Z data
[~, contPlot] = contour(xMesh1, pMesh1, Z_initial, 25, 'LineColor', "k"); % 100 contour levels in black

% Initialize the time display
timeText = text(0.95, 0.95, '', ...
    'Units', 'normalized', ...          % Position relative to axes (0 to 1)
    'HorizontalAlignment', 'right', ... % Align text to the right
    'VerticalAlignment', 'top', ...     % Align text to the top
    'FontSize', 25, ...                  % Set desired font size
    'FontWeight', 'bold', ...            % Make text bold for visibility
    'Color', 'w');                       % Set text color to black
uistack(timeText, 'top');                 % Ensure text is on top

% Main simulation loop
for n = 1:length(WigSSE(1,1,:))
    % Retrieve the current simulation time from plotSpanSSE
    t = plotSpanSSE(n);

    % Update the pcolor plot with new data for the current time step
    set(f, 'CData', WigSSE(:,:,n));

    % Compute contour data based on function H
    CdatCont = arrayfun(@(x, y) Hinit([x; y], n), xMesh1, pMesh1);

    % ---------------------- Dynamic Contour Update ----------------------

    % Delete existing contour plot to prevent overlap
    delete(contPlot);

    % Recompute and plot new contours with updated data
    [~, contPlot] = contour(xMesh1, pMesh1, CdatCont, 25, 'LineColor', "k");

    % Optionally, adjust contour levels here if needed
    % For example, to specify levels:
    % levels = linspace(min(CdatCont(:)), max(CdatCont(:)), 100);
    % [~, contPlot] = contour(xMesh, pMesh, CdatCont, levels, 'LineColor', "k");

    % ---------------------- End of Dynamic Contour Update -------------------

    % Update the experimental line plot with new data point
    xData = get(expLine, 'XData');
    yData = get(expLine, 'YData');
    set(expLine, 'XData', [xData, expXSSE(n)], 'YData', [yData, expPSSE(n)]);

    % ---------------------- Update Time Display ----------------------

    % Update the text object with the current time, formatted to two decimal places
    set(timeText, 'Interpreter', 'latex', ...
        'String', sprintf('$N = $ %.2f, $\\langle \\hat\\theta_{\\phi^{+}}\\rangle = $ %.2f', t, CtSSE(n)));


    % Ensure the time text stays on top of other plot elements
    uistack(timeText, 'top');

    % ---------------------- End of Time Display Update -------------------

    % Refresh the plot to reflect updates
    drawnow;

    % Optional: Capture the frame for a flipbook or video
    flipbook(n) = getframe(gcf);
end

% Save video
writerObj = VideoWriter(sprintf('WignerMarkovSSEVideo_lambda_%.2g_Hubble_%.2g',lambda,Hb));
% writerObj = VideoWriter(sprintf('WignerMarkovSSEVideo_Hubble_%.2g',Hb));

writerObj.FrameRate = nFramesSSE/30;
open(writerObj);
writeVideo(writerObj, flipbook);
close(writerObj);

close all;

%% Panels SSE


[~, idxOrig] = min(abs(plotSpanSSE- targets(1)), [], 2);   % idx(i) indexes p nearest to t(i)
[~, idxZero] = min(abs(plotSpanSSE- targets(2)), [], 2);   % idx(i) indexes p nearest to t(i)
WigImagesSSE(:,:,1)=WigSSE(:,:, idxOrig);
WigImagesSSE(:,:,2)=WigSSE(:,:, idxZero);
WigImagesSSE(:,:,3)=WigSSE(:,:, end);


% ---------- (a) N_e = -0.75 ----------
figA = figure('Position', [100 100 800 600]);  % wider figure for visibility
axA = axes(figA); hold(axA,'on');
hA = imagesc(axA, x, p, WigImagesSSE(:,:,1));
set(hA,'Interpolation','bilinear');   % or 'bicubic'
colormap(axA,'parula');
contour(axA, xMesh1, pMesh1, Hgrid_at(targets(1)), 25, 'LineColor', 'k');
xlim(axA,[-Xview Xview]); ylim(axA,[-Xview Xview]); axis(axA,'square'); box(axA,'on'); caxis(axA,cax);
xticks(linspace(-Xview,Xview,9))
colorbar(axA,'FontSize',18);
expLine = plot(expXSSE(1:idxOrig), expPSSE(1:idxOrig), 'w-', 'LineWidth', 1); % White line with specified width
text(axA, 0.98, 0.98, fmtText(targets(1), CtSSE(idxOrig)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(axA, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(axA,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(axA,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(axA, '(g)','Interpreter','latex','FontSize',30);
box on
exportgraphics(figA,'SSEwigner_panel_a_Ne_-0p75.pdf','ContentType','vector');

% ---------- (b) N_e = 0 ----------
figB = figure('Position', [100 100 800 600]);  % wider figure for visibility
axB = axes(figB); hold(axB,'on');
hB = imagesc(axB, x, p, WigImagesSSE(:,:,2));
set(hB,'Interpolation','bilinear');   % or 'bicubic'
colormap(axB,'parula');
contour(axB, xMesh1, pMesh1, Hgrid_at(targets(2)), 25, 'LineColor', 'k');
xlim(axB,[-Xview Xview]); ylim(axB,[-Xview Xview]); axis(axB,'square'); box(axB,'on'); caxis(axB,cax);
xticks(linspace(-Xview,Xview,9))
colorbar(axB,'FontSize',18);
expLine = plot(expXSSE(1:idxZero), expPSSE(1:idxZero), 'w-', 'LineWidth', 1); % White line with specified width
text(axB, 0.98, 0.98, fmtText(targets(2), CtSSE(idxZero)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(axB, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(axB,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(axB,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(axB, '(h)','Interpreter','latex','FontSize',30);
box on
exportgraphics(figB,'SSEwigner_panel_b_Ne_0.pdf','ContentType','vector');

% ---------- (c) N_e = +1.5 ----------
figA = figure('Position', [100 100 800 600]);  % wider figure for visibility
axA = axes(figA); hold(axA,'on');
hC = imagesc(axA, x, p, WigImagesSSE(:,:,3));
set(hC,'Interpolation','bilinear');   % or 'bicubic'
colormap(axA,'parula');
contour(axA, xMesh1, pMesh1, Hgrid_at(targets(3)), 25, 'LineColor', 'k');
xlim(axA,[-Xview Xview]); ylim(axA,[-Xview Xview]); axis(axA,'square'); box(axA,'on'); caxis(axA,cax);
xticks(linspace(-Xview,Xview,9))
colorbar(axA,'FontSize',18);
expLine = plot(expXSSE(1:end), expPSSE(1:end), 'w-', 'LineWidth', 1); % White line with specified width
text(axA, 0.98, 0.98, fmtText(targets(3),CtSSE(end)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(axA, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(axA,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(axA,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(axA, '(i)','Interpreter','latex','FontSize',30);
box on
exportgraphics(figA,'SSEwigner_panel_c_Ne_+0p75.pdf','ContentType','vector');


%% Create a figure with specified position and aspect ratio
close all;

clear flipbook
WigLind = RhoWigner(RhoLindPos, x, p, hbar);

figureHandle = figure('Position', [100, 100, 800, 800 * aspectRatio(2) / aspectRatio(1)]);
Hinit = @(Z,n) (exp(3*plotSpanLind(n))*(-(mu^2/2)*Z(1)^2+(mu*alpha3/6)*Z(1)^3+(alpha4/24)*Z(1)^4)+0.5*Z(2)^2*exp(-3*plotSpanLind(n)))/Hb;

% Initialize pcolor plot with the first set of data
f = pcolor(xMesh, pMesh, WigLind(:,:,1));
xlim([-Xview Xview]);
ylim([-Xview Xview]);
shading interp
colormap("parula")  % Changed colormap to 'parula' for perceptually uniform colors

% Set color axis limits
clim([-0.05 0.15])  % Ensure this range covers both oscillator and inverted oscillator regimes
xticks(linspace(-Xview,Xview,9))
% Add a colorbar with increased font size
colorbar('FontSize', 14);

title( sprintf(...
    'Lindblad Evolution, $\\lambda = %.2f$, $\\tilde \\mu = %.2f$', ...
    lambda, mu/Hb), ...
    'FontSize', 20, 'Interpreter', 'latex' );
% title( sprintf(...
%     'Lindblad Evolution, $H = %.2f$', Hb), ...
%     'FontSize', 20, 'Interpreter', 'latex' );
% xlabel('$\phi$', 'FontSize', 25, 'Interpreter', 'latex');
% ylabel('$\pi_{\phi}$', 'FontSize', 25, 'Interpreter', 'latex');

% Customize axis properties for better aesthetics
set(gca, 'FontSize', 16, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
axis square  % Use square axes for equal aspect ratio

% Hold the current plot to allow multiple graphics objects
hold on;

% Initialize an empty line plot for experimental data
expLine = plot(NaN, NaN, 'w-', 'LineWidth', 1); % White line with specified width

% Initialize contour plot with initial Z data
[~, contPlot] = contour(xMesh1, pMesh1, Z_initial, 25, 'LineColor', "k"); % 100 contour levels in black

% Initialize the time display
timeText = text(0.95, 0.95, '', ...
    'Units', 'normalized', ...          % Position relative to axes (0 to 1)
    'HorizontalAlignment', 'right', ... % Align text to the right
    'VerticalAlignment', 'top', ...     % Align text to the top
    'FontSize', 25, ...                  % Set desired font size
    'FontWeight', 'bold', ...            % Make text bold for visibility
    'Color', 'w');                       % Set text color to black
uistack(timeText, 'top');                 % Ensure text is on top

% Main simulation loop
for n = 1:length(WigLind(1,1,:))
    % Retrieve the current simulation time from plotSpanLind
    t = plotSpanLind(n);

    % Update the pcolor plot with new data for the current time step
    set(f, 'CData', WigLind(:,:,n));

    % Compute contour data based on function H
    CdatCont = arrayfun(@(x, y) Hinit([x; y], n), xMesh1, pMesh1);

    % ---------------------- Dynamic Contour Update ----------------------

    % Delete existing contour plot to prevent overlap
    delete(contPlot);

    % Recompute and plot new contours with updated data
    [~, contPlot] = contour(xMesh1, pMesh1, CdatCont, 25, 'LineColor', "k");

    % Optionally, adjust contour levels here if needed
    % For example, to specify levels:
    % levels = linspace(min(CdatCont(:)), max(CdatCont(:)), 100);
    % [~, contPlot] = contour(xMesh, pMesh, CdatCont, levels, 'LineColor', "k");

    % ---------------------- End of Dynamic Contour Update -------------------

    % Update the experimental line plot with new data point
    xData = get(expLine, 'XData');
    yData = get(expLine, 'YData');
    set(expLine, 'XData', [xData, expXLind(n)], 'YData', [yData, expPLind(n)]);

    % ---------------------- Update Time Display ----------------------

    % Update the text object with the current time, formatted to two decimal places
    set(timeText, 'Interpreter', 'latex', ...
        'String', sprintf('$N = $ %.2f, $\\langle \\hat\\theta_{\\phi^{+}}\\rangle = $ %.2f', t, CtLind(n)));


    % Ensure the time text stays on top of other plot elements
    uistack(timeText, 'top');

    % ---------------------- End of Time Display Update -------------------

    % Refresh the plot to reflect updates
    drawnow;

    % Optional: Capture the frame for a flipbook or video
    flipbook(n) = getframe(gcf);
end

% Save video
writerObj = VideoWriter(sprintf('WignerMarkovLindVideo_lambda_%.2g_Hubble_%.2g',lambda,Hb));

writerObj.FrameRate = nFramesLind/30;
open(writerObj);
writeVideo(writerObj, flipbook);
close(writerObj);


%% Panels Lindblad



[~, idxOrig] = min(abs(plotSpanLind- targets(1)), [], 2);   % idx(i) indexes p nearest to t(i)
[~, idxZero] = min(abs(plotSpanLind- targets(2)), [], 2);   % idx(i) indexes p nearest to t(i)
WigImagesLind(:,:,1)=WigLind(:,:, idxOrig);
WigImagesLind(:,:,2)=WigLind(:,:, idxZero);
WigImagesLind(:,:,3)=WigLind(:,:, end);


% ---------- (a) N_e = -0.75 ----------
figA = figure('Position', [100 100 800 600]);  % wider figure for visibility
axA = axes(figA); hold(axA,'on');
hA = imagesc(axA, x, p, WigImagesLind(:,:,1));
set(hA,'Interpolation','bilinear');   % or 'bicubic'
colormap(axA,'parula');
contour(axA, xMesh1, pMesh1, Hgrid_at(targets(1)), 25, 'LineColor', 'k');
xlim(axA,[-Xview Xview]); ylim(axA,[-Xview Xview]); axis(axA,'square'); box(axA,'on'); caxis(axA,cax);
xticks(linspace(-Xview,Xview,9))
colorbar(axA,'FontSize',18);
expLine = plot(expXLind(1:idxOrig), expPLind(1:idxOrig), 'w-', 'LineWidth', 1); % White line with specified width
text(axA, 0.98, 0.98, fmtText(targets(1), CtLind(idxOrig)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(axA, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(axA,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(axA,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(axA, '(a)','Interpreter','latex','FontSize',30);
box on
exportgraphics(figA,'Lindwigner_panel_a_Ne_-0p75.pdf','ContentType','vector');

% ---------- (b) N_e = 0 ----------
figB = figure('Position', [100 100 800 600]);  % wider figure for visibility
axB = axes(figB); hold(axB,'on');
hB = imagesc(axB, x, p, WigImagesLind(:,:,2));
set(hB,'Interpolation','bilinear');   % or 'bicubic'
colormap(axB,'parula');
contour(axB, xMesh1, pMesh1, Hgrid_at(targets(2)), 25, 'LineColor', 'k');
xlim(axB,[-Xview Xview]); ylim(axB,[-Xview Xview]); axis(axB,'square'); box(axB,'on'); caxis(axB,cax);
xticks(linspace(-Xview,Xview,9))
colorbar(axB,'FontSize',18);
expLine = plot(expXLind(1:idxZero), expPLind(1:idxZero), 'w-', 'LineWidth', 1); % White line with specified width
text(axB, 0.98, 0.98, fmtText(targets(2), CtLind(idxZero)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(axB, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(axB,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(axB,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(axB, '(b)','Interpreter','latex','FontSize',30);
box on
exportgraphics(figB,'Lindwigner_panel_b_Ne_0.pdf','ContentType','vector');

% ---------- (c) N_e = +1.5 ----------
figA = figure('Position', [100 100 800 600]);  % wider figure for visibility
axA = axes(figA); hold(axA,'on');
hC = imagesc(axA, x, p, WigImagesLind(:,:,3));
set(hC,'Interpolation','bilinear');   % or 'bicubic'
colormap(axA,'parula');
contour(axA, xMesh1, pMesh1, Hgrid_at(targets(3)), 25, 'LineColor', 'k');
xlim(axA,[-Xview Xview]); ylim(axA,[-Xview Xview]); axis(axA,'square'); box(axA,'on'); caxis(axA,cax);
xticks(linspace(-Xview,Xview,9))
colorbar(axA,'FontSize',18);
expLine = plot(expXLind(1:end), expPLind(1:end), 'w-', 'LineWidth', 1); % White line with specified width
text(axA, 0.98, 0.98, fmtText(targets(3),CtLind(end)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(axA, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(axA,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(axA,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(axA, '(c)','Interpreter','latex','FontSize',30);
box on
exportgraphics(figA,'Lindwigner_panel_c_Ne_+0p75.pdf','ContentType','vector');
% 
% 
% %
% 
close all;
% 
tEnd = cputime - tStart


save('workspaceSingles')

%% Sonification
% ===================== Tunables =====================
BSound = 200;           % number of lowest eigenstates to sonify
% f0     = 65.406;       % base note (Hz), C
f0     = 20;       % base note (Hz), C

fs     = 44100;        % audio sample rate

% ===================== FFmpeg check =====================
[ff_ok, ~] = system('ffmpeg -version');
if ff_ok ~= 0
    error('FFmpeg is not installed or not on PATH.');
end

% ===================== Render 30s audio for each trajectory =====================
[InstAudioPath,  InstAudioDur]  = write_simple_sonification('Inst',  RhoInst, plotSpanInst,  Xhat, Phat, Hb, mu, alpha3,  alpha4,BSound, f0, fs);
[HamAudioPath,  HamAudioDur]  = write_simple_sonification('Ham',  RhoSchrodinger, plotSpanHam,  Xhat, Phat, Hb, mu, alpha3, alpha4, BSound, f0, fs);
[SSEAudioPath,  SSEAudioDur]  = write_simple_sonification('SSE',  RhoSSE,         plotSpanSSE,  Xhat, Phat, Hb, mu, beta3,  beta4,BSound, f0, fs);
[LindAudioPath, LindAudioDur] = write_simple_sonification('Lind', RhoLind,        plotSpanLind, Xhat, Phat, Hb, mu, alpha3, alpha4,BSound, f0, fs);
% (Each *_AudioDur will be 30.000 s)

% ===================== Merge each with its 30s Wigner video =====================
InstVideoPath= sprintf('WignerGroundStateVideo.avi');
HamVideoPath  = sprintf('WignerSchrodingerVideo_Hubble_%.2g.avi', Hb);
SSEVideoPath  = sprintf('WignerMarkovSSEVideo_lambda_%.2g_Hubble_%.2g.avi', lambda, Hb);
LindVideoPath = sprintf('WignerMarkovLindVideo_lambda_%.2g_Hubble_%.2g.avi', lambda, Hb);

InstMP4Out  = 'Inst_EigenbasisSimple_synced.mp4';
HamMP4Out  =sprintf( 'Ham_EigenbasisSimple_synced_Hubble_%.2g.mp4', Hb);
SSEMP4Out  =sprintf( 'SSE_EigenbasisSimple_lambda_%.2g_Hubble_%.2g.mp4', lambda, Hb);
LindMP4Out = sprintf('Lind_EigenbasisSimple_lambda_%.2g_Hubble_%.2g.mp4', lambda, Hb);

% Mux without forcing -t; videos are 30 s and WAVs are 30 s → perfect sync
merge_audio_video_ffmpeg(InstVideoPath,  InstAudioPath,  InstMP4Out,  InstAudioDur);
merge_audio_video_ffmpeg(HamVideoPath,  HamAudioPath,  HamMP4Out,  HamAudioDur);
merge_audio_video_ffmpeg(SSEVideoPath,  SSEAudioPath,  SSEMP4Out,  SSEAudioDur);
merge_audio_video_ffmpeg(LindVideoPath, LindAudioPath, LindMP4Out, LindAudioDur);

disp('All sonifications rendered and merged with videos (30 s each).')

%% Spectrograms
for n=1:nFramesSSE
    RhoSSE(:,:,n)=PsiSSE(:,n)*PsiSSE(:,n)';
end
% --- compute energies & occupations ---
[~, EnergiesHam, HamOcc]  = HamiltonianEigenrep2(RhoSchrodinger, plotSpanHam,  Xhat, Phat, Hb, mu, alpha3,alpha4);
[~, EnergiesSSE, SSEOcc]  = HamiltonianEigenrep2(RhoSSE,         plotSpanSSE,  Xhat, Phat, Hb, mu, beta3,beta4);
[~, EnergiesLind, LindOcc]= HamiltonianEigenrep2(RhoLind,        plotSpanLind, Xhat, Phat, Hb, mu, alpha3,alpha4);

%% plotting

% --- make the three figures in order: Ham, SSE, Lind ---
makePanel('EnergyLines_panelHam.pdf',  plotSpanHam,  EnergiesHam,  HamOcc,  '(a)');
makePanel('EnergyLines_panelSSE.pdf',  plotSpanSSE,  EnergiesSSE,  SSEOcc,  '(a)');
makePanel('EnergyLines_panelLind.pdf', plotSpanLind, EnergiesLind, LindOcc, '(c)');
%% Exit
% run('MarkovCoarseGrainedLindbladScript.m')

exit


%% --- panel maker (local function) ---
function makePanel(figName, plotSpan, Energies, Occ, panelLabel)
    fig = figure('Position',[100 100 2400 600]);  %#ok<NASGU>
    ax  = axes; hold(ax,'on'); set(gcf,'Renderer','opengl');

    % draw colored lines into ax
    [~, cb] = plotEnergyLinesByOccupation( ...
        plotSpan, Energies, Occ, ...
        'CutoffFinal',    1e-20, ...
        'MinOccForColor', 1e-10, ...
        'LineWidth',      2, ...
        'Colormap',       'jet', ...
        'Parent',         ax);

    % your layout/limits
    set(ax,'YLim',[1e0 1e3],'YLimMode','manual');
    set(ax,'XLim',[min(plotSpan) max(plotSpan)],'XLimMode','manual');
    axis(ax,'normal');
    set(ax,'PlotBoxAspectRatioMode','auto','DataAspectRatioMode','auto');
    ax.XTick = linspace(-2,1, 7);
    % place axes & colorbar nicely
    drawnow;
    ax.Units = 'normalized'; cb.Units = 'normalized';
    ax.Position = [0.08 0.15 0.78 0.78];
    cb.Position(1) = 0.88;

    % styling
    set(ax, 'FontSize',20, 'LineWidth',1.5, 'Box','on', 'TickLabelInterpreter','latex');
    xlabel(ax,'$N$','Interpreter','latex','FontSize',25);
    ylabel(ax,'$K_s$','Interpreter','latex','FontSize',25);
    title(ax, panelLabel,'Interpreter','latex','FontSize',30);
    set(cb,'FontSize',18);

    % export vector PDF
    exportgraphics(gcf, figName, 'ContentType','image');
end


%% ============================== HELPER FUNCTIONS (BOTTOM) ==============================
function [ax, cb, cLimits] = plotEnergyLinesByOccupation(plotSpan, EnergiesOut, EnergyOcc, varargin)
% Plot energy-vs-time lines with color = ADDITIVE occupation in a log-energy bin.
% Returns:
%   ax       : axes used (your 'Parent' if provided)
%   cb       : colorbar handle
%   cLimits  : [cmin cmax] in log10 units used for CLim
%
% NOTE: This function does NOT set x/y limits, aspect, or axis 'tight/square'.
%       You control layout/limits outside.

% ------------ options ------------
p = inputParser;
addParameter(p,'IdxCut',[],@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(p,'CutoffFinal',1e-3,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p,'MinOccForColor',1e-12,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p,'LineWidth',1.5,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p,'Colormap','turbo');
addParameter(p,'NumColorBins',500,@(x)isnumeric(x)&&isscalar(x)&&x>=2);
addParameter(p,'ColorERange',[],@(x)isnumeric(x)&&numel(x)==2);
addParameter(p,'Parent',[],@(x) isempty(x) || ishghandle(x,'axes'));
parse(p,varargin{:});
idxCut       = p.Results.IdxCut;
cutFinal     = p.Results.CutoffFinal;
minOcc       = p.Results.MinOccForColor;
lw           = p.Results.LineWidth;
cmapName     = p.Results.Colormap;
NbColor      = p.Results.NumColorBins;
colorERange  = p.Results.ColorERange;
ax           = p.Results.Parent;

% ------------ sizes & sanity ------------
[b, N] = size(EnergiesOut);
assert(isequal(size(EnergyOcc), [b N]), 'EnergyOcc must be size(EnergiesOut).');
t = plotSpan(:).';

% target axes
if isempty(ax)
    fig = figure('Position',[100 100 800 600]); %#ok<NASGU>
    ax  = axes; 
end
hold(ax,'on'); set(gcf,'Renderer','opengl');

% auto idxCut
if isempty(idxCut)
    idxCut = find(EnergyOcc(:,end) >= cutFinal, 1, 'last');
    if isempty(idxCut), idxCut = min(b,1); end
end
idxCut = min(max(1, round(idxCut)), b);

% ------------ log-energy binning for COLOR aggregation ------------
Eall = EnergiesOut(1:idxCut,:);
if isempty(colorERange)
    EminPos = min(Eall(Eall>0),[],'all');
    if isempty(EminPos), error('All energies are non-positive; need positive energies for log bins.'); end
    EmaxVal = max(Eall,[],'all');
    if ~(EmaxVal > EminPos), EmaxVal = EminPos*(1+1e-9); end
    Emin = 0.999*EminPos;
    Emax = EmaxVal;
else
    Emin = colorERange(1);
    Emax = colorERange(2);
    if Emin <= 0, error('ColorERange(1) must be > 0 for log-spaced bins.'); end
end
Eedges = logspace(log10(Emin), log10(Emax), NbColor+1).';

% ------------ additive color values per level/time ------------
Csum = zeros(idxCut, N);   % summed occupation for each level/time via its bin
for n = 1:N
    E_n   = Eall(:,n);
    occ_n = EnergyOcc(1:idxCut, n);
    idx_n = discretize(E_n, Eedges);              % 1..NbColor or NaN

    S = accumarray(idx_n(~isnan(idx_n)), occ_n(~isnan(idx_n)), [NbColor,1], @sum, 0);
    valid = ~isnan(idx_n);
    Csum(valid,n) = S(idx_n(valid));
end
Csum = max(Csum, minOcc);  % floor to avoid -Inf in log10

% global color limits in log10
cmin = log10(min(Csum(:)));
cmax = log10(max(Csum(:)));
cLimits = [cmin cmax];

% ------------ draw colored “ribbons” (edges only) ------------
for k = 1:idxCut
    Ek = EnergiesOut(k,:);            % 1 x N
    Ck = log10(Csum(k,:));            % log-color per time

    X2 = [t; t];
    Y2 = [Ek; Ek];
    Z2 = zeros(2, N);
    C2 = [Ck; Ck];

    surf(ax, X2, Y2, Z2, C2, ...
        'FaceColor','none', ...
        'EdgeColor','interp', ...
        'LineWidth', lw, ...
        'EdgeAlpha', 1); 
end

% axes scale only (no limits/aspect here)
set(ax,'YScale','log');

% colormap + colorbar
try, colormap(ax, cmapName); catch, colormap(ax,'parula'); end
caxis(ax, cLimits);
cb = colorbar(ax);
pmin = floor(cmin); pmax = ceil(cmax); if pmax < pmin, pmax = pmin; end
ticks = pmin:pmax;
if numel(ticks) < 2
    ticks = [cmin cmax];
    cb.Ticks = ticks;
    cb.TickLabels = arrayfun(@(v) sprintf('10^{%.2g}', v), ticks, 'uni', 0);
else
    cb.Ticks = ticks;
    cb.TickLabels = arrayfun(@(p) sprintf('10^{%d}', p), ticks, 'uni', 0);
end
ylabel(cb, '$P(K_s)$','Interpreter','latex');

end



function [outRho, EnergiesOut, EnergyOcc] = HamiltonianEigenrep2(RhoIn, plotSpan, Xhat, Phat, Hb, mu, alpha3,alpha4)
% HamiltonianEigenrep2
% Inputs:
%   RhoIn (bSize x bSize x N): density matrices in the fixed basis at each time
%   plotSpan (1 x N): times (or e-folds Ne) to evaluate
%   Xhat, Phat: operators in the fixed basis
%   Hb, Height, Separation: scalars used in the Hamiltonian
%
% Outputs:
%   outRho   (bSize x bSize x N): density matrices in the instantaneous energy basis (sorted by energy)
%   EnergiesOut (bSize x N): energies at each time (shifted so min=1, matching your original behavior)
%   EnergyOcc   (bSize x N): vector of energy occupations (populations) at each time

bSize = size(RhoIn, 1);
N     = length(plotSpan);

EnergiesOut = zeros(bSize, N);
EnergyOcc   = zeros(bSize, N);
outRho      = zeros(bSize, bSize, N);

Id  = eye(bSize);
H0     = @(Ne) (0.5/Hb) * (Phat^2 * exp(-3*Ne) ...
    + (-(mu^2/2)*Xhat^2 + (mu*alpha3/6)*Xhat^3 + (alpha4/24)*Xhat^4) * exp(3*Ne));

for n = 1:N
    % Diagonalize instantaneous Hamiltonian
    [V, D] = eig(H0(plotSpan(n)));
    evals  = real(diag(D));

    % Sort energies ascending, and reorder eigenvectors consistently
    [evals_sorted, idx] = sort(evals, 'ascend');
    V = V(:, idx);

    % Store (normalized) energies like in your original code
    EnergiesOut(:, n) = evals_sorted - min(evals_sorted) + 1;

    % Transform rho into the energy eigenbasis
    TempRho = V' * RhoIn(:, :, n) * V;
    TempRho = 0.5 * (TempRho + TempRho');         % enforce Hermiticity
    TempRho = TempRho / trace(TempRho);           % normalize

    outRho(:, :, n) = TempRho;

    % Occupations = diagonal elements in energy basis
    occ = real(diag(TempRho));

    % Numerical cleanup: clip tiny negative values and renormalize
    occ(occ < 0 & occ > -1e-12) = 0;
    s = sum(occ);
    if s ~= 0
        occ = occ / s;
    end

    EnergyOcc(:, n) = occ;
end
end


function [audioPath, total_duration] = write_simple_sonification(basename, RhoSeq, plotSpan, ...
        Xhat, Phat, Hb, mu, alpha3,alpha4, BSound, f0, fs)

    % 1) Instantaneous eigen-representation with TIME TRACKING + PHASE FIX
    %    (labels are consistent across frames; phases are continuous)
    [outRho, Ener] = HamiltonianEigenrep(RhoSeq, plotSpan, Xhat, Phat, Hb, mu, alpha3,alpha4);

    % 2) Render 30 s stereo audio using the tracked sequences
    total_duration = 30.0;  % seconds (fixed target)
    audioLR = render_simple_binaural_from_plotspan(outRho, Ener, plotSpan, f0, fs, total_duration, BSound);

    % 3) Save WAV
    audioPath = [basename '_EigenbasisSimple.wav'];
    audiowrite(audioPath, audioLR, fs);

    fprintf('Wrote %s (%.3f s)\n', audioPath, total_duration);
end

function audio_stereo = render_simple_binaural_from_plotspan(RhoSeq, ESeq, plotSpan, f0, fs, total_dur, BSound)
    % RhoSeq: (b x b x T), in a CONSISTENT tracked eigenbasis, Hermitian, trace~1
    % ESeq  : (b x T), tracked energies (E0=1 so f_n = E_n * f0)
    % plotSpan: (1 x T) physical timestamps (nonuniform)
    % total_dur: target audio length in seconds (30)

    [bFull, T] = size(ESeq);
    b = min(BSound, bFull);

    % Frequencies for the b lowest tracked bands (ground=1 -> f0)
    F = ESeq(1:b,:) * f0;  % (b x T)

    if T < 2
        audio_stereo = zeros(0,2); 
        return;
    end

    % Allocate per-interval sample counts from plotSpan
    dN   = diff(plotSpan(:));                 
    span = plotSpan(end) - plotSpan(1);       
    frac = dN / max(span, eps);               
    L    = round(frac * total_dur * fs);      
    need = total_dur*fs - sum(L);
    if need ~= 0
        % Distribute rounding residue across intervals to avoid a single big seam
        sgn = sign(need);
        idxs = 1:(T-1);
        k = 1;
        while need ~= 0
            L(idxs(k)) = L(idxs(k)) + sgn;
            need = need - sgn;
            k = k + 1;
            if k > numel(idxs), k = 1; end
        end
    end
    L = max(L,1);

    starts = [1; 1 + cumsum(L(1:end-1))];
    Ntot   = sum(L);
    audio_stereo = zeros(Ntot,2);

    % Lower-triangle voices (ket->Left, bra->Right), no double count
    [Li,Ri] = ndgrid(1:b,1:b);
    mask = (Li >= Ri);
    kIdx = Li(mask); lIdx = Ri(mask);
    M = numel(kIdx);

    % Magnitudes & phases from tracked density matrices
    Abs = abs(RhoSeq(1:b,1:b,:));
    Arg = angle(RhoSeq(1:b,1:b,:));

    % Unwrap phases over time and reference to first frame
    for k = 1:b
        for l = 1:b
            Arg(k,l,:) = unwrap(squeeze(Arg(k,l,:)));
        end
    end
    Arg = Arg - Arg(:,:,1);

    % Synthesize interval by interval
    t_offset = 0;  % seconds
    for t = 1:T-1
        Lt = L(t);
        alpha = linspace(0,1,Lt);                   % interpolation parameter

        % Interpolate magnitudes and phases
        m0 = Abs(:,:,t);   m1 = Abs(:,:,t+1);
        p0 = Arg(:,:,t);   p1 = Arg(:,:,t+1);
        m0 = m0(mask);     m1 = m1(mask);
        p0 = p0(mask);     p1 = p1(mask);

        mags = m0.*(1-alpha) + m1.*alpha;           % (M x Lt)
        phis = p0.*(1-alpha) + p1.*alpha;           % (M x Lt)

        % Interpolate per-voice frequencies within the interval
        fL0 = F(kIdx, t);   fL1 = F(kIdx, t+1);
        fR0 = F(lIdx, t);   fR1 = F(lIdx, t+1);
        fLs = fL0.*(1-alpha) + fL1.*alpha;          % (M x Lt)
        fRs = fR0.*(1-alpha) + fR1.*alpha;          % (M x Lt)

        % Absolute time vector for this interval
        t_frame = (0:Lt-1)/fs + t_offset;           % 1 x Lt
        tM = repmat(t_frame, M, 1);                 % (M x Lt)

        % Binaural additive synthesis
        frameL = mags .* sin(2*pi.*fLs.*tM + phis);
        frameR = mags .* sin(2*pi.*fRs.*tM - phis);

        % Write into output
        i0 = starts(t); i1 = i0 + Lt - 1;
        audio_stereo(i0:i1,1) = sum(frameL,1).';
        audio_stereo(i0:i1,2) = sum(frameR,1).';

        t_offset = t_offset + Lt/fs;
    end

    % Normalize
    mx = max(abs(audio_stereo(:)));
    if mx > 0, audio_stereo = 0.9 * audio_stereo / mx; end
end

function merge_audio_video_ffmpeg(videoPath, audioPath, outPath, ~)
    if ~isfile(videoPath), error('Video file not found: %s', videoPath); end
    if ~isfile(audioPath), error('Audio file not found: %s', audioPath); end

    cmd = sprintf(['ffmpeg -y -i "%s" -i "%s" ' ...
                   '-map 0:v:0 -map 1:a:0 ' ...
                   '-c:v libx264 -c:a aac -b:a 192k -shortest "%s"'], ...
                   videoPath, audioPath, outPath);
    disp(['Executing: ' cmd]);
    [st, out] = system(cmd);
    if st ~= 0
        fprintf(2, 'FFmpeg error output:\n%s\n', out);
        error('Failed to merge audio and video to %s', outPath);
    else
        fprintf('Merged audio+video -> %s\n', outPath);
    end
end

function [outRho, EnergiesOut] = HamiltonianEigenrep(RhoIn, plotSpan, Xhat, Phat, Hb,mu, alpha3,alpha4)
% TIME-TRACKED instantaneous eigen-representation.
% - Sorts the FIRST frame by ascending energy.
% - For subsequent frames: assigns modes by maximum overlap with previous eigenbasis.
% - Fixes eigenvector phases so overlaps with previous frame are real-positive.
%
% Inputs:
%   RhoIn:    (b x b x T) density matrices in Fock basis
%   plotSpan: (1 x T)
%   Xhat, Phat, Hb, Height, Separation: operators/params for H(N)
%
% Outputs (both are in the tracked instantaneous eigenbasis at each t):
%   outRho:       (b x b x T)
%   EnergiesOut:  (b x T), with the initial frame ascending & labels tracked

    bSize = size(RhoIn,1);
    T     = length(plotSpan);

    outRho       = zeros(bSize,bSize,T);
    EnergiesOut  = zeros(bSize,T);

    Id  = eye(bSize);
    H0     = @(Ne) (0.5/Hb) * (Phat^2 * exp(-3*Ne) ...
    + (-(mu^2/2)*Xhat^2 + (mu*alpha3/6)*Xhat^3 + (alpha4/24)*Xhat^4) * exp(3*Ne));
    % --- Frame 1: eigendecomposition, sort by ascending energy
    [V_prev, D_prev] = eig(full(H0(plotSpan(1))));
    E_prev = real(diag(D_prev));
    [E_prev_sorted, perm0] = sort(E_prev, 'ascend');
    V_prev = V_prev(:, perm0);

    % Phase gauge for the very first frame: make largest component real-positive
    for i = 1:bSize
        [~,ix] = max(abs(V_prev(:,i)));
        s = V_prev(ix,i) / max(abs(V_prev(ix,i)), eps);
        V_prev(:,i) = V_prev(:,i) * conj(s) / max(abs(s), eps);
    end

    % Project first-frame rho
    EnergiesOut(:,1) = E_prev_sorted - min(E_prev_sorted) + 1;
    R1 = V_prev' * RhoIn(:,:,1) * V_prev;
    R1 = 0.5*(R1+R1'); R1 = R1 / max(trace(R1), eps);
    outRho(:,:,1) = R1;

    % --- Subsequent frames: track by maximum overlap + phase fix
    for t = 2:T
        Ht = H0(plotSpan(t));
        [V_curr, D_curr] = eig(full(Ht));
        E_curr = real(diag(D_curr));

        % Compute overlaps O_ij = <v_i(prev) | v_j(curr)>
        O = V_prev' * V_curr;

        % Assignment: maximize |O| (greedy if matchpairs unavailable)
        perm = assign_max_overlap(O);   % size bSize, maps i(prev) -> perm(i) in curr

        % Permute current eigenvectors and energies
        V_curr = V_curr(:, perm);
        E_curr = E_curr(perm);

        % Phase gauge: make diagonal overlaps real-positive
        diagOv = diag(V_prev' * V_curr);
        ph = exp(-1i * angle(diagOv + (diagOv==0)));  % avoid NaN if zero
        V_curr = V_curr * diag(ph);

        % Project Rho into the TRACKED curr basis
        Rt = V_curr' * RhoIn(:,:,t) * V_curr;
        Rt = 0.5*(Rt+Rt'); Rt = Rt / max(trace(Rt), eps);

        EnergiesOut(:,t) = E_curr - min(E_curr) + 1;
        outRho(:,:,t)    = Rt;

        % Advance
        V_prev = V_curr;
    end
end

function perm = assign_max_overlap(O)
% Heuristic maximum-overlap assignment:
% - If Optimization Toolbox is available, use matchpairs on -abs(O)
% - Else, fall back to a greedy row-wise argmax with column exclusion
    try
        % Requires Optimization Toolbox (R2016b+)
        cost = -abs(O);
        pairs = matchpairs(cost, -Inf);  % returns [rowIdx colIdx]
        perm = zeros(1,size(O,1));
        perm(pairs(:,1)) = pairs(:,2);
    catch
        % Greedy fallback (works well when tracking is near-diagonal)
        n = size(O,1);
        perm = zeros(1,n);
        used = false(1,n);
        for i = 1:n
            [~, jlist] = sort(abs(O(i,:)),'descend');
            j = jlist(find(~used(jlist),1,'first'));
            if isempty(j), j = find(~used,1,'first'); end
            perm(i) = j;
            used(j) = true;
        end
    end
end


function Hn = hermite_poly(n, x)
% Recursively compute physicists' Hermite polynomials H_n(x)
if n == 0
    Hn = ones(size(x));
elseif n == 1
    Hn = 2*x;
else
    H0nes = ones(size(x));
    H1 = 2*x;
    for k = 2 : n
        Hn = 2*x .* H1 - 2*(k-1)*H0nes;
        H0nes = H1;
        H1 = Hn;
    end
end
end
