#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
template<class T>
void print(const vector<T> vec){
    cout<<"{ ";
    for (auto x : vec) cout<<x<<", ";
    cout<<" }"<<endl;
}
vector<vector<int> > threeSum(vector<int>& nums){
    sort(nums.begin(), nums.end());
    print(nums);
    vector<vector<int>> results;
    int sum = 0;
    for (int i=0;i<nums.size();++i){
        if(i>0 && nums[i]==nums[i-1]) continue; //去重
        //if(i>0 && nums[i]==nums[i+1]) continue; //错误, 会漏掉情况
        int L = i+1, R = nums.size()-1;
        while(L<R){
            sum = nums[i]+nums[L]+nums[R];
            cout<<sum<<endl;
            cout<<nums[i]<<" "<<nums[L]<<" "<<nums[R]<<endl;
            if(sum==0){
                results.push_back(vector<int>({nums[i], nums[L], nums[R]}));
                while(L<R && nums[L]==nums[L+1]) ++L; //去重
                while(L<R && nums[R]==nums[R-1]) --R; //去重
                ++L; --R;
            }
            else if(sum>0) R--;
            else if(sum<0) L++;
        }

    }
    return results;
}

int main(){
    vector<int> nums = {-1, 0, 1, 2, -1, -4};
    auto rsts = threeSum(nums);
    cout<<rsts.size()<<endl;
    for( auto x: rsts) print(x);

}
