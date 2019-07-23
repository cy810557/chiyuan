#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

void print(const vector<int> &vi){
    cout<<"{ ";
    for (auto x : vi) cout<<x<<", ";
    cout<<" }"<<endl;
}
vector<vector<int> > addThree(vector<int> nums){
    sort(nums.begin(), nums.end());
    print(nums);
    int _size = nums.size();
    vector<vector<int> > results_ = {};
    if(nums[0]<=0 && nums[_size-1]>0){  //整个数组同号，则无解
        for (int i=0;i!=_size;++i){  //选出队伍的第一个人，即组织者
            if (nums[i] > 0) break;  //组织者不能是高手
            int first = i+1; //初始化菜鸟
            int last = _size-1;  //初始化高手
            while(1){
                if(first>=last || nums[i]*nums[last]>0) break; //终止条件
                int result = nums[i]+nums[first]+nums[last];
                print(vector<int>({nums[i], nums[first], nums[last]}));
                cout<<i<<" "<<first<<" "<<last<<endl;
                if(result==0){
                    results_.push_back(vector<int>({nums[i], nums[first], nums[last]}));
                    //print(results_[results_.size()-1]);
                }
                if(result>0){  //实力太强，需要高手往左移动一位（若移动后相同则继续）
                    while(first<last) {
                        --last;
                        if (nums[last]!=nums[last-1]) break;
                    }
                }else
                {
                    while(first<last) {
                        ++first;
                        if (nums[first]==nums[first+1]) break;
                    }
                }
            }
        }
    }
    return results_;
}

int main(){
    vector<int> vi={-1,0,1,2,-1,4};
    auto results_ = addThree(vi);
    cout<<results_.size()<<endl;
    cout<<"======"<<endl;
    for(auto x:results_) print(x);
}