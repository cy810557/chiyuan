#include "vector_utils.cpp"
#include<iostream>

int main(){
    vector<int> vi={1,3,5,7,9,8,10};
    // vector slice
    print(vi);
    print(vector<int>(&vi[2], &vi[5]));
    print(vector<int>(vi.begin()+3, vi.end()-2));

    //auto rst = find(vi.begin(), vi.end(),4);
    //cout<<"found: "<<*(rst)<<endl;
    vi.pop_back();
    reverse(vi.begin(), vi.end());
    print(vi);
    cout<<vi.size()<<endl;
    vi = {};
    cout<<vi.size()<<endl;
    vi.insert(vi.end(), 9, -6);
    print(vi);

    //sorted(key=lambda x)
    vector<vector<int>> intervals;
    intervals.push_back(vector<int>({2,3}));
    intervals.push_back(vector<int>({1,4}));
    intervals.push_back(vector<int>({6,8}));
    print(intervals);
    stable_sort(intervals.begin(), intervals.end(), 
        [](const vector<int> &v1, const vector<int> &v2){return v1[0]<=v2[0];});
    cout<<"After sorting: "<<endl;
    print(intervals);
    auto it = intervals.begin()+1;
    intervals.erase(it, it+1);
    cout<<"After erasing: "<<endl;
    print(intervals);

    //test constructer
    cout<<"test constructer: "<<endl;
    vector<int> vcons(10,1);
    print(vcons);


}
