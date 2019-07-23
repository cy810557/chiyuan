#include<iostream>
#include<vector>
#include<algorithm>
#include "../vector_utils.cpp"
using namespace std;

void go_through(vector<int> &vi, int target, vector<vector<int>> &results, vector<int> &path, vector<int>::iterator begin){
    // 终止条件: target=0
    if(target==0) results.push_back(path);
    //每个节点遍历分支得到子节点时，不是对vi中所有元素遍历，应该只遍历比父节点分支数值大的分支,以避免重复
    for (;begin<vi.end();++begin){  
        int residual = target - *begin;
        cout<<"residual: "<<residual<<endl;
        print(path);
        path.push_back(*begin);
        if (residual < 0) {path.pop_back();break;}
        go_through(vi, residual, results, path, begin);
        path.pop_back(); //key!用于回溯到上一级节点
    }
}

vector<vector<int> > combination_sum(vector<int> &vi, int target){
    sort(vi.begin(), vi.end());
    vector<vector<int> > results;
    vector<int> path;
    go_through(vi, target, results, path, vi.begin());
    return results;
}

int main(){
    vector<int> vi = {8,7,4,3};
    auto results = combination_sum(vi, 11);
    /**
    vector<int> vi = {2,3,5};
    auto results = combination_sum(vi, 8);
    **/
    cout<<"results: "<<endl;
    for (auto result:results) print(result);
}

