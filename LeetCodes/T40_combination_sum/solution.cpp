#include <iostream>
#include<vector>
#include<set>
#include<algorithm>
#include "../vector_utils.cpp"
using namespace std;

template <typename T>
void del_duplicate(vector<T> &v){
    set<T> s(v.begin(), v.end());
    v.assign(s.begin(), s.end());
}

void go_through(vector<int> &vi, int target, vector<int> &route, vector<vector<int>> &results, vector<int>::iterator beg){
    if (target==0) results.push_back(route);
    cout<<"target: "<<target<<" ";
    //print(vector<int>(beg, vi.end()));
    //print(route);
    for (;beg!=vi.end();++beg){
        route.push_back(*beg); //出错！每遍历一个子节点首先要更新route, 遍历退出时再pop掉
        int residual = target - *beg;
        if (residual < 0) {route.pop_back(); break;}
        else{
            go_through(vi, residual, route, results, beg+1);
            route.pop_back();
        }
    }
}

vector<vector<int> > combination_sum(vector<int> &vi, int target){
    sort(vi.begin(), vi.end());
    vector<vector<int>> results;
    vector<int> route;
    go_through(vi, target, route, results, vi.begin());
    return results;
}

int main()
{
    vector<int> vi = {10, 1, 2, 7, 6, 1, 5};
    int target = 8;
    auto results = combination_sum(vi, target);
    cout<<endl;
    cout<<"=======results========="<<endl;
    //set<vector<int>>s(results.begin(), results.end());
    //results.assign(s.begin(), s.end());
    del_duplicate(results);
    for( result:results ){ print(result); }
    return 0;
}

