#include <iostream>
#include <vector>
#include "../vector_utils.cpp"
#include <stack>
#include <algorithm>
using namespace std;

void merge_backward(vector<vector<int>> &intervals, vector<vector<int>>::iterator last){
    if (last-intervals.begin()<1) return ;
    //若当前区间和前一个区间符合合并条件，则扩展前面区间的右侧边界，同时将两区间左侧边界最小值赋给前面区间。最后删除当前区间。
    if ((*last)[0]<=(*(last-1))[1]) {    
        (*(last-1))[1] = (*last)[1];
        ((*(last-1))[0] = min((*(last-1))[0], (*last)[0]));
        intervals.erase(last, last+1); 
    } 
    --last;
    cout<<"merge once: ";
    print(intervals);
    merge_backward(intervals, last);
}

vector<vector<int>> merge(vector<vector<int>>& intervals){
    auto last = intervals.end()-1;
    //排序：使得每个区间的首元素按从小到大的顺序排列
    stable_sort(intervals.begin(), intervals.end(),
        [](vector<int> v1, vector<int> v2){return top(v1)<=top(v2);});
    merge_backward(intervals, last);
    return intervals;
}

int main(){
    vector<vector<int>> intervals;
    intervals.push_back(vector<int>({2,3}));
    intervals.push_back(vector<int>({1,4}));
    print(intervals);
    auto results = merge(intervals);
    cout<<"result: ";
    print(results);
}
