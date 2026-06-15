my_stu_number = []; %手动输入自己的学号
target_length = 289;
origin = zeros(1, target_length);
current_idx = 1;  % 当前位置索引
counter = 1;

while current_idx <= target_length
    for i = 1:12
        power_val = my_stu_number(i) ^ counter;
        
        % 转换为字符串并拆分为数字
        digits = num2str(power_val) - '0';
        
        % 添加到数组
        for d = digits
            if current_idx <= target_length
                origin(current_idx) = d;
                current_idx = current_idx + 1;
            else
                break;
            end
        end
        
        if current_idx > target_length
            break;
        end
    end
    counter = counter + 1;
end


% 构造矩阵A
A = reshape(origin, [17, 17])';

disp('生成的17阶方阵:');
disp(A);
fprintf('矩阵维度: %d×%d\n', size(A));

% 保存到文件
save('A.mat', 'A');
fprintf('已保存到 matrix_17x17.mat\n');




%%开始计算
load A.mat

% 第一题：求A的行列式
det_A = det(A);
fprintf("行列式：\n");
disp(det_A);

% 第二题：求A的标准阶梯型矩阵；求AX=0通解；尝试求满秩分解
H = rref(A);
null_space = null(A, 'r');

fprintf("解空间通解：\n")
disp(null_space);

function [F, G] = full_rank_rref(A)
    % A - 输入矩阵
    % F - 满秩分解 m * r
    % G - 满秩分解 r * n

    % 计算行最简形
    [R, pivot_cols] = rref(A);
    
    % 提取主元列
    F = A(:, pivot_cols);
    
    % 提取非零行
    r = length(pivot_cols);  % 矩阵的秩
    G = R(1:r, :);
end
[F, G] = full_rank_rref(A);
fprintf("满秩分解(A = F x G)：\n");
disp(F);
disp(G);

% 第三题：列向量组的极大线性无关组；求施密特标准正交基
function max_independent_set = max_independent_column_arrays(A)
    [R, pivots] = rref(A);

    % 极大线性无关组
    max_independent_set = A(:, pivots);
end

max_independent_set = max_independent_column_arrays(A);
fprintf("极大线性无关组：\n");
disp(max_independent_set);

function Q = gram_schmidt_full(A)
    % A - 输入矩阵
    % Q - 标准正交矩阵
    % R - 上三角矩阵
    
    [m, n] = size(A);
    Q = zeros(m, n);
    R = zeros(n, n);
    
    for j = 1:n
        % 第j个向量
        v = A(:, j);
        
        % 减去在前面正交向量上的投影
        for i = 1:j-1
            R(i, j) = Q(:, i)' * v;
            v = v - R(i, j) * Q(:, i);
        end
        
        % 计算范数
        R(j, j) = norm(v);
        
        if R(j, j) > eps
            Q(:, j) = v / R(j, j);
        else
            Q(:, j) = zeros(m, 1);
            warning('第%d个向量过小或线性相关', j);
        end

    end
end

orthonormal_basis = gram_schmidt_full(max_independent_set);
fprintf("施密特正交化后标准正交基：\n");
disp(orthonormal_basis);

% 第四题：求A + AT的正负惯性指数
X = A + A';
evaluator = 1e-10;
% 解出X的特征值
eigenvalues = eig(X);

positive = sum(eigenvalues > evaluator);
negative = sum(eigenvalues < -evaluator);
fprintf("A + AT的正负惯性指数：\n");
fprintf("正惯性指数：%d\n", positive);
fprintf("负惯性指数：%d\n", negative);

%%
% 输出结果
save("输出.mat","det_A","H","null_space","F","G","max_independent_set","orthonormal_basis","positive","negative")