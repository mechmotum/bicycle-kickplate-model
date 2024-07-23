function [xd, nf] = eval_rhs(t, x, r, p)

    q = x(1:8);
    u = x(9:16);
    f = x(17:20);

    [A, b] = eval_dynamic_eqs(q, u, f, r, p);
    xplus = A\b;

    xd = [u; xplus(1:12)];
    nf = xplus(13:14);

end
