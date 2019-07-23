#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

void print(const vector<int> &vi){
    cout<<"{ ";
    for (auto x : vi) cout<<x<<", ";
    cout<<" }"<<endl;
}

vector<vector<int>> threeSum(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> results_={};
    int _size = nums.size();
    if (nums[0]<=0 && nums[_size-1]>=0){
        for(int i=0;i!=_size;++i){
            if (nums[i] > 0) break;
            int first = i+1;
            int last = _size-1;
            while(1){
                if (first>=last || nums[i]*nums[last]>0) break;
                int result = nums[i] + nums[first] + nums[last];
                if (result==0){
                    results_.push_back(vector<int>({nums[i], nums[first], nums[last]}));
                }
                if (result>=0){
                    do{
                        --last;
                        }while(first<last && nums[last]==nums[last-1]);
                    }
                else {  //result <= 0
                    do{
                        ++first;  
                        }while(first<last && nums[first]==nums[first+1]);
                     }
            }
            do{
                ++i;
            }while(nums[i]==nums[i+1]);  //去除重复元素
        }
    }
    return results_;
}

int main(){
    vector<int> vi={-1,0,1,2,-1,4};
    auto results_ = threeSum(vi);
    cout<<results_.size()<<endl;
    for(auto x:results_) print(x);
}