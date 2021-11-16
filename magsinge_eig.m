% Matrix B
[B, rows, cols, entries] = mmread('B.mtx');

% Matrix A
[A, rows, cols, entries] = mmread('A.mtx');

% Solver call
nev = 5;
ncv = 50;
tau = 0 + 0i;
[V, D] = eigs(A, B, nev, tau, 'SubspaceDimension', ncv);

% Save of eigenmodes and parameters simulations
% Eigenvalues
txt1 = cat(2, real(diag(D)), imag(diag(D)));
title_txt1 = 'Eigenval.txt';
save(title_txt1, 'txt1', '-ascii');
% Eigenvectors
txt21 = real(V.');	txt22 = imag(V.');
title_txt21 = 'Real_Eigenvec.txt';
title_txt22 = 'Imag_Eigenvec.txt';
save(title_txt21, 'txt21', '-ascii');
save(title_txt22, 'txt22', '-ascii');