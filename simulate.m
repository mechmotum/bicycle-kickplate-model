p.c_af = 11.46;
p.c_ar = 11.46;
p.c_pf = 0.573;
p.c_pr = 0.573;
p.c_maf = 0.01;
p.c_mar = 0.01;
p.c_mpf = 0.01;
p.c_mpr = 0.01;
p.d1 = 0.9534570696121849;
p.d2 = 0.2676445084476887;
p.d3 = 0.03207142672761929;
p.g = 9.81;
p.ic11 = 7.178169776497895;
p.ic22 = 11.0;
p.ic31 = 3.8225535938357873;
p.ic33 = 4.821830223502103;
p.id11 = 0.0603;
p.id22 = 0.12;
p.ie11 = 0.05841337700152972;
p.ie22 = 0.06;
p.ie31 = 0.009119225261946298;
p.ie33 = 0.007586622998470264;
p.if11 = 0.1405;
p.if22 = 0.28;
p.l1 = 0.4707271515135145;
p.l2 = -0.47792881146460797;
p.l3 = -0.00597083392418685;
p.l4 = -0.3699518200282974;
p.mc = 85.0;
p.md = 2.0;
p.me = 4.0;
p.mf = 3.0;
p.rf = 0.35;
p.rr = 0.3;
p.s_yf = 0.15;
p.s_yr = 0.15;
p.s_zf = 0.15;
p.s_zr = 0.15;

% initial coordinates
q_vals = [0.0, 0.0, 0.0, 0.0, nan, 0.0, 1e-10, 0.0];
[initial_pitch_angle, ~] = solve_for_pitch(q_vals(4), q_vals(7), p.d1, p.d2, p.d3, p.rf, p.rr);
q_vals(5) = initial_pitch_angle;
q_vals

% initial speeds
initial_speed = 4.6;  % m/s
u_vals = [nan, nan, 0.0, 0.5, nan, -initial_speed/p.rr, 0.0, -initial_speed/p.rf];

p_vals = [p.c_af, p.c_ar, p.c_pf, p.c_pr, p.c_maf, p.c_mar, p.c_mpf, ...
    p.c_mpr, p.d1, p.d2, p.d3, p.g, p.ic11, p.ic22, p.ic31, p.ic33, p.id11, ...
    p.id22, p.ie11, p.ie22, p.ie31, p.ie33, p.if11, p.if22, p.l1, p.l2, ...
    p.l3, p.l4, p.mc, p.md, p.me, p.mf, p.rf, p.rr, p.s_yf, p.s_yr, p.s_zf, ...
    p.s_zr];

% TODO : yd0 should be properly handled, but should mostly be zero
[A, b] = eval_dep_speeds(q_vals, u_vals([3, 4, 6, 7, 8]), 0.0, [p.d1, p.d2, p.d3, p.rf, p.rr])
u_vals([1, 2, 5]) = A\b;
u_vals

f_vals = [0.0, 0.0, 0.0, 0.0];

initial_conditions = [q_vals, u_vals, f_vals]

r_vals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

eval_rhs(0.0, initial_conditions', r_vals', p_vals')

fps = 100;
duration = 6;
t0 = 0.0;
tf = t0 + duration;
times = linspace(t0, tf, duration*fps);

f_anon = @(t, x) eval_rhs(t, x, r_vals, p_vals);
[ts, xs] = ode45(f_anon, times, initial_conditions);

subplot(211)
plot(ts, xs(:, 4))
ylabel('Angle [rad]')
subplot(212)
plot(ts, xs(:, 12))
ylabel('Angular Rate [rad]')
xlabel('Time [s]')
