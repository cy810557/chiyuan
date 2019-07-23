#include<vector>
#include<iostream>
#include<algorithm>
using namespace std;
//算法：维护一个辅助的vector；同时从后向前遍历所给向量nums，寻找“顺序对”;
//      初始阶段先将nums最后一个元素pop到辅助vector temp中；
//      进入循环：只要nums不为空，判断nums最后一个元素（顶端）是否大于等于temp顶端元素；
//           若是，则将nums顶端元素pop到temp顶端，继续循环；
//           若否，则已经找到第一个顺序对。此时将nums顶端元素和temp中大于该元素的最小元素交换位置, 退出循环
//      最后：在剩下的nums基础上逐次将temp中存放的元素按照由小到大的顺序取回。注意到temp中元素存放的顺序已经是从小到大了。
//

void print(const vector<int> &vi){
    cout<<"DEBUG: { ";
    for (auto x: vi) cout<<x<<", ";
    cout<<" }"<<endl;
}

void print(const vector<vector<int>::iterator> &vi){
    cout<<"DEBUG: { ";
    for (auto x: vi) cout<<*x<<", ";
    cout<<" }"<<endl;
}

static int pop(vector<int> &vi){
    int last = *(vi.end()-1);
    vi.pop_back();
    return last;
}

static int& top(vector<int> &vi){
    return *(vi.end()-1);
}

int swap_lager(vector<int> &vi, int n){
    auto beg = vi.cbegin();
    for (int i=0; i<vi.size();++i){
        if(vi[i]>n) {
            int temp = vi[i];
            vi[i] = n;
            return temp;
        }
    }
    return -1;
}

void nextPermutation(vector<int>& nums) {
    vector<int> temp={pop(nums)};
    while(!nums.empty()){
        if(top(nums) >= top(temp)){
            temp.push_back(pop(nums));
        } else {
            // 交换nums顶部元素和temp中大于nums顶部元素的最小元素
            top(nums)  = swap_lager(temp, top(nums));
            break;
        }
    }
    for(int i=0;i<temp.size();++i){
        nums.push_back(temp[i]);
    }
}

int main(){
    //vector<int> vi = {1,2,3,5,4};
    //vector<int> vi = {2,3,1};
    vector<int> vi = {1,3,2};
    print(vi);
    nextPermutation(vi);
    print(vi);
}
