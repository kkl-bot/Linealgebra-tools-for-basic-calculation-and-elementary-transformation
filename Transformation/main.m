%随机生成一个10×8阶的整数矩阵，编程或用Matlab命令化成阶梯形矩阵。
random_seed = 212; % Set a random seed for reproducibility
rng(random_seed); % Initialize the random number generator with the specified seed
row = 10;
col = 8;
random_matrix = randi(500, row, col);
disp('Random Matrix:');
disp(random_matrix);
A = rref(random_matrix);
disp(A);
%% 

%写出Householder变换求解矩阵QR分解的计算机算法程序，并求A的QR分解。
function [v, beta] = householder_vector(x)
    % 输入: x 列向量
    % 输出: v 满足 H = I - beta*v*v' 的向量, 且 v(1)=1
    %       beta 模
    % 准备函数，用于构造H1、H2……Hn
    
    n = length(x);
    sigma = x(2:end)' * x(2:end);          % 下三角部分的平方和
    v = x;
    v(1) = 1;
    
    if sigma == 0 && x(1) >= 0
        beta = 0;                          % x 已经是对角化方向
        v = zeros(n,1);
        v(1) = 1;
        return;
    end
    
    mu = sqrt(x(1)^2 + sigma);
    if x(1) <= 0
        v(1) = x(1) - mu;
    else
        v(1) = -sigma / (x(1) + mu);
    end
    
    beta = 2 * v(1)^2 / (sigma + v(1)^2);
    v = v / v(1);
end

function [Q, R] = householder_qr(A)
    [m, n] = size(A);
    R = A;
    Q = eye(m);
    
    for k = 1:min(m-1, n)
        % 提取当前列从 k 行到末行的子向量
        x = R(k:end, k);
        [v,beta] = householder_vector(x);
        
        % 构造反射向量
        v_full = zeros(m, 1);
        v_full(k:end) = v;
        
        for j = k:n
            t = beta * (v_full' * R(:, j));
            R(:, j) = R(:, j) - t * v_full;
        end
        
        % 累积 Q 时使用 Q = Q * H，使得最终 A = Q * R
        Q = Q - beta * (Q * v_full) * v_full';
    end
    
    R = triu(R);
end
B = [3 14 9;6 43 3;6 22 15;];
[Q,R] = householder_qr(B);
disp(Q);
disp(R);
disp(Q*R);
[Q_ref, R_ref] = qr(B);
disp(Q_ref);
disp(R_ref);
disp(Q*Q.')
%% 

%写出Givens变换求解矩阵QR分解的计算机算法程序，并求A的QR分解。
B = [2 2 1;0 2 2;2 1 2;];

function [c, s] = givens_rotation(a, b)
    % 准备函数，用于计算Givens旋转的余弦和正弦
    if b == 0
        c = 1;
        s = 0;
    else
        r = sqrt(a^2 + b^2);
        c = a / r;
        s = b / r;
    end
end

function G = apply_givens(A, c, s, i, j)
    % 将 Givens 旋转作用于矩阵 A 的第 i 行和第 j 行，
    % 影响从第 k 列到最后一列的所有元素。
    % 输入:
    %   A - 待操作矩阵
    %   c, s - Givens 旋转参数
    %   i, j - 要旋转的两行索引 (i < j)
    [m,n] = size(A);
    G = eye(m,n);
    G(i,i) = c;
    G(j,j) = c;
    G(i,j) = -s;
    G(j,i) = s;
    % 已经废弃，可以直接对矩阵操作节省内存
end

function [Q, R] = givens_qr(A)
    % 使用 Givens 旋转计算 QR 分解 A = Q * R
    % 返回正交矩阵 Q 和上三角矩阵 R
    [m, n] = size(A);
    R = A;
    Q = eye(m);
    
    for col = 1:n
        for row = m:-1:(col+1)     % 从最后一行开始，向上消去
            if R(row, col) ~= 0
                i = col;
                j = row;
                a = R(col, col);
                b = R(row, col);
                [c, s] = givens_rotation(a, b);
                for k = col:n
                    R_ik = R(i, k);
                    R_jk = R(j, k);
                    R(i, k) = c * R_ik + s * R_jk;
                    R(j, k) = -s * R_ik + c * R_jk;
                end
                for k = 1:m
                    Q_ki = Q(k, i);
                    Q_kj = Q(k, j);
                    Q(k, i) = c * Q_ki + s * Q_kj;
                    Q(k, j) = -s * Q_ki + c * Q_kj;
                end
            end
        end
    end
    
    % 确保 R 的上三角部分严格（下三角已为零）
    R = triu(R);
end
[Q_givens, R_givens] = givens_qr(B);
disp(Q_givens);
disp(R_givens);
disp(Q_givens*R_givens);
disp(Q*R);