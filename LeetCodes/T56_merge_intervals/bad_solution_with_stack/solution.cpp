#include <iostream>
#include <vector>
#include "../vector_utils.cpp"
#include <stack>
using namespace std;

vector<vector<int>> merge(vector<vector<int>>& intervals){
    vector<vector<int>> new_interval={};
    stack<vector<int>> svec;
    stable_sort(intervals.begin(), intervals.end(),
        [](const vector<int> &v1, const vector<int> &v2){return v1[0]<=v2[0];});

    svec.push(*intervals.begin());
    
    for (auto beg=intervals.begin()+1; beg!=intervals.end();++beg){
        if(top(svec.top())>=(*beg)[0]){
            cout<<"merge once"<<endl;
            print(svec.top());
            print(*beg);
            top(svec.top()) = max(top(*beg), top(svec.top()));
            cout<<"After merging: "<<top(*beg)<<" ";
            print(svec.top());
        }
        else{
            new_interval.push_back(svec.top());
            svec.pop();
            svec.push(*beg);
            }
        }
     if (!svec.empty()){
        new_interval.push_back(svec.top());
        svec.pop();
        }
    return new_interval;
}

int main(){
    vector<vector<int>> intervals;
    intervals.push_back(vector<int>({2,5}));
    intervals.push_back(vector<int>({1,5}));
    //intervals.push_back(vector<int>({8,10}));
    print(intervals);
    auto results = merge(intervals);
    print(results);
}
