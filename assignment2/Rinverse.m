function [R, Ri] = Rinverse(n)
    A = rand(n); 
    A = A + diag(sum(abs(A)));
    R = triu(A);
    Ri = compute_inverse(R);
    fprintf('Error in computing inverse: %e\n', ...
             norm(Ri*R-eye(size(R))));
return

function Ri = compute_inverse(R)
    n = size(R,1);
    if (n < 16)
        Ri = inv(R); 
    else
        n1 = round(n/2);
        Ri = R; 
        Ri(1:n1,1:n1) = compute_inverse(Ri(1:n1,1:n1));
        Ri(n1+1:n,n1+1:n) = compute_inverse(Ri(n1+1:n,n1+1:n)); 
        Ri(1:n1,n1+1:n) = - Ri(1:n1,1:n1) * Ri(1:n1,n1+1:n) ...
                             * Ri(n1+1:n,n1+1:n);
    end
return
