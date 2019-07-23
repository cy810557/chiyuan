#include<iostream>
#include<vector>
#include "../vector_utils.cpp"
using namespace std;

void go_through(vector<int> &vi, int target, vector<int> &result){
    cout<<"target: "<<target<<" ";
    print(vi);
    if(vi.empty()) return;
    int last = top(vi);
    if(target<vi[0]) {result={};return;}  // target小于vi剩下的所有元素
    if(target%last==0) {
        result.insert(result.end(), target/last, last);
        return;
    } else {
        if(vi.size()==1) {
            result={};
            return;
        }
        else{
            result.push_back(pop(vi));
            go_through(vi, target-last, result);
        }
    }
}

vector<vector<int> > combination_sum(vector<int> &vi, int target){
    vector<vector<int> > results;
    vector<int> result;
    for(auto end=vi.end();end!=vi.begin();--end){
        cout<<"Loop once!"<<endl;
        result = {};
        auto vcurr = vector<int>(vi.begin(), end); //每次遍历删去vi的末尾元素，拷贝一份送入go_through
        go_through(vcurr, target, result);
        if(result.size()>0) results.push_back(result);
    }
    return results;
}

int main(){
    /**
    vector<int> vi = {2,3,6,7};
    auto results = combination_sum(vi, 7);
    **/
    vector<int> vi = {2,3,5};
    auto results = combination_sum(vi, 8);
    cout<<"results: "<<endl;
    for (auto result:results) print(result);
}

