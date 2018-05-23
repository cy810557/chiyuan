%%  T2
rng(923) %随机数种子
n = 1e3;
data = num2cell([unifrnd(-10, 10, n, 3), randi(1e2, [n, 1])],1);
inputdata = table(data{:}, 'VariableNames', {'X', 'Y', 'Z', 'Group'});
head(inputdata)
tic
r = compute(inputdata);
toc
%%
function result = compute(input)
input = sortrows(input,4);
input = cell2mat(table2cell(input)); %*****注意：较高端操作***** 将table转换为mat，从而方便进行处理
group_num = unique(input(:,4),'rows'); %unique函数用于去重，这里按列方向去重
X = cell( [1, length(group_num)] ); %预定义一个空的cell来保存分组的号码及其中的坐标点
%%  生成各个组的矩阵X（j）
for j = 1:size(group_num);
    count = 0;
    for i = 1:size(input,1)
        if (input(i,4)==group_num(j,1));
            count = count + 1 ;
            X{j}(count,:) = input(i,:); % ******注意了：高端操作！******
            %          Group_dis = pdist(A,(:,1:3));
        end
    end
end
nG =length(X);
result = cell(nG*(nG-1)/2,2); %%注意：预定义一定大小的cell
% result ={};
count2=0;
%% 接下来遍历任意两个分组中的点之间的距离
for m = 1:nG
    min_matrix = zeros(size(X{m},1),1); %预定义
    for n = m+1:nG  %注意，如果从1开始的话，计算距离会出现重复
        dist_2rows = zeros(size(X{n},1),1);
        for p = 1: size(X{m},1) % 遍历第m个矩阵的每一行
            %         dist_row_group = min(pdist([X{m}(p,:); X{n}])); %第m个矩阵的第p行与第n个矩阵组成。×××
            %注意，这里是错的，要求的是第一个点与其他所有点的距离，而不是所有点之间的距离。
            for q = 1:size(X{n},1) %遍历第n个矩阵的每一行
                p_2rows = [ X{m}(p,:); X{n}(q,:)];
                dist_2rows(q) = pdist(p_2rows);  %两个点之间的距离
            end
            min_matrix(p,1) = min(dist_2rows); %距离矩阵，接收第m个小组的第p个点与第n个小组所有点之间的最小值
        end
        min_groups = min(min_matrix);
        %         fprintf('第%d组和第%d组之间的最小距离为%d\n',m,n,min_groups);
        count2 = count2 + 1;
        %         result{count2,1} = {['第',num2str(m), '组和第',num2str(n),'组之间的距离'],min_groups};
        result{count2,1} = ['第',num2str(m), '组和第',num2str(n),'组之间的距离'];
        result{count2,2} = min_groups;
    end
end
end